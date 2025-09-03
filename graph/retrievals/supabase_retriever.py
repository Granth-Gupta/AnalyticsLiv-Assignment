from typing import List, Optional
from pathlib import Path

from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from supabase import Client

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
    # UUID-safe, type-agnostic: DELETE WHERE id IS NOT NULL [web:98][web:164]
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

    if table_is_empty(supabase_client, SALES_TABLE):
        csv_loader = CSVLoader(
            file_path=CSV_PATH,
            source_column="Year",
            csv_args={
                "delimiter": ",",
                "quotechar": '"',
                "fieldnames": ["Month", "Year", "Total_Sales", "Transactions"],
            },
            encoding="utf-8",  # helpful with BOM/locale issues [web:156]
        )
        docs = csv_loader.load()
        if not docs:
            raise RuntimeError(f"No rows loaded from CSV: {CSV_PATH}")
        SupabaseVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            client=supabase_client,
            table_name=SALES_TABLE,
            query_name=QUERY_FN_SALES,
            chunk_size=500,
        )

    if table_is_empty(supabase_client, FAQ_TABLE):
        faq_docs = TextLoader(str(Path(TXT_PATH)), encoding="utf-8").load()  # explicit encoding [web:146][web:148]
        if not faq_docs:
            raise RuntimeError(f"No text loaded from TXT: {TXT_PATH}")
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=70)
        texts = splitter.split_documents(faq_docs)
        if not texts:
            raise RuntimeError("Text splitter produced no FAQ chunks")
        SupabaseVectorStore.from_documents(
            documents=texts,
            embedding=embeddings,
            client=supabase_client,
            table_name=FAQ_TABLE,
            query_name=QUERY_FN_FAQ,
            chunk_size=500,
        )

    # Open vector stores
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

    csv_retriever = vectorstore_csv.as_retriever(search_kwargs={"k": 8})
    faq_retriever = vectorstore_faq.as_retriever(search_kwargs={"k": 8})

    return csv_retriever, faq_retriever, vectorstore_csv, vectorstore_faq

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
            docs_all.append(Document(page_content=r.get("content", "") or "", metadata=r.get("metadata") or {}))
        if len(rows) < page_size:
            break
        start += page_size

    return RunnableLambda(lambda _: docs_all)
