from langchain_community.document_loaders import CSVLoader, TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from dotenv import load_dotenv
import chromadb
import os

load_dotenv()

def collection_exists(name: str, client) -> bool:
    try:
        client.get_collection(name)
        return True
    except Exception:
        return False

def data_retriever(PERSIST_DIR: str, CSV_PATH: str, TXT_PATH: str):
    persist_dir_path = Path(PERSIST_DIR)
    persist_dir_path.mkdir(parents=True, exist_ok=True)

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    client = chromadb.PersistentClient(path=PERSIST_DIR)

    if not collection_exists("sales_collection", client):
        csv_loader = CSVLoader(CSV_PATH, encoding="windows-1252")
        csv_docs = csv_loader.load()

        Chroma.from_documents(
            documents=csv_docs,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
            collection_name="sales_collection",
        ).persist()

    if not collection_exists("faq_collection", client):
        faq_docs = []
        txt_path = Path(TXT_PATH)
        faq_docs = TextLoader(str(txt_path), encoding="utf-8").load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        texts = text_splitter.split_documents(faq_docs)

        print("Total Chunks created from text file: ",len(texts))

        Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
            collection_name="faq_collection",
        ).persist()

    db1 = Chroma(
        persist_directory=PERSIST_DIR,
        collection_name="sales_collection",
        embedding_function=embeddings,
    )
    db2 = Chroma(
        persist_directory=PERSIST_DIR,
        collection_name="faq_collection",
        embedding_function=embeddings,
    )

    csv_retriever = db1.as_retriever()
    faq_retriever = db2.as_retriever()

    return csv_retriever, faq_retriever
