from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain_community.vectorstores import FAISS, SupabaseVectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv
from supabase import Client

load_dotenv()

SALES_TABLE = "sales_collection"
FAQ_TABLE = "faq_collection"
QUERY_FN_FAQ = "match_documents_faq"
QUERY_FN_SALES = "match_documents_sales"

def table_is_empty(client: Client, table_name: str) -> bool:
    try:
        resp = client.table(table_name).select("id", count="exact").limit(1).execute()
        return (getattr(resp, "count", 0) or 0) == 0
    except Exception:
        return True

def delete_all_rows(client: Client, table_name: str) -> int:
    # UUID-safe, type-agnostic: DELETE WHERE id IS NOT NULL
    resp = client.table(table_name).delete().not_.is_("id", None).execute()
    data = getattr(resp, "data", None)
    return len(data) if isinstance(data, list) else 0

def data_retriever(supabase_client: Client, CSV_PATH: Optional[str], TXT_PATH: Optional[str]):
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    csv_ok = bool(CSV_PATH and Path(CSV_PATH).is_file())
    txt_ok = bool(TXT_PATH and Path(TXT_PATH).is_file())

    # Clear both tables only when both inputs are present
    if csv_ok and txt_ok:
        delete_all_rows(supabase_client, SALES_TABLE)
        delete_all_rows(supabase_client, FAQ_TABLE)

    # Seed sales if empty
    if table_is_empty(supabase_client, SALES_TABLE) and csv_ok:
        csv_loader = CSVLoader(
            file_path=CSV_PATH,
            source_column="Year",
            csv_args={"delimiter": ",", "quotechar": '"', "fieldnames": ["Month", "Year", "Total_Sales", "Transactions"]},
            encoding="utf-8",
        )
        sales_docs = csv_loader.load()
        if not sales_docs:
            raise RuntimeError(f"No rows loaded from CSV: {CSV_PATH}")
        SupabaseVectorStore.from_documents(
            documents=sales_docs,
            embedding=embeddings,
            client=supabase_client,
            table_name=SALES_TABLE,
            query_name=QUERY_FN_SALES,
            chunk_size=500,
        )

    # Seed FAQ if empty
    if table_is_empty(supabase_client, FAQ_TABLE) and txt_ok:
        faq_docs_src = TextLoader(str(Path(TXT_PATH)), encoding="utf-8").load()
        if not faq_docs_src:
            raise RuntimeError(f"No text loaded from TXT: {TXT_PATH}")
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=70)
        faq_docs = splitter.split_documents(faq_docs_src)
        if not faq_docs:
            raise RuntimeError("Text splitter produced no FAQ chunks")
        SupabaseVectorStore.from_documents(
            documents=faq_docs,
            embedding=embeddings,
            client=supabase_client,
            table_name=FAQ_TABLE,
            query_name=QUERY_FN_FAQ,
            chunk_size=500,
        )

    # 1) Open Supabase vector stores (persistent store)
    vectorstore_csv = SupabaseVectorStore(
        client=supabase_client,
        embedding=embeddings,
        table_name=SALES_TABLE,
        query_name=QUERY_FN_SALES,
    )
    vectorstore_faq = SupabaseVectorStore(
        client=supabase_client,
        embedding=embeddings,
        table_name=FAQ_TABLE,
        query_name=QUERY_FN_FAQ,
    )

    csv_retriever_sb = vectorstore_csv.as_retriever(search_kwargs={"k": 8})
    faq_retriever_sb = vectorstore_faq.as_retriever(search_kwargs={"k": 8})

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    retrievalQA_sales = RetrievalQA.from_llm(llm=llm, retriever=csv_retriever_sb)  # [1][2]
    retrievalQA_faq = RetrievalQA.from_llm(llm=llm, retriever=faq_retriever_sb)

    return {
        "retrievalQA_sales": retrievalQA_sales,
        "retrievalQA_faq": retrievalQA_faq,
    }

def all_csv_docs_retriever(supabase_client: Client) -> RunnableLambda:
    page_size = 100
    docs_all: List[Document] = []
    start = 0
    while True:
        end = start + page_size - 1
        resp = supabase_client.table(SALES_TABLE).select("content, metadata").range(start, end).execute()
        rows = getattr(resp, "data", None) or []
        if not rows:
            break
        for r in rows:
            docs_all.append(
                Document(
                    page_content=r.get("content", "") or "",
                    metadata=r.get("metadata") or {}
                )
            )
        if len(rows) < page_size:
            break
        start += page_size

    return RunnableLambda(lambda _: docs_all)