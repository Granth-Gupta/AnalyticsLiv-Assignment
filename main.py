import os
import time
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Form, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from Supabase.client import supabase_client
from graph.graph import llm_chain
from graph.retrievals.supabase_retriever import data_retriever, all_csv_docs_retriever
from graph.Prompts.prompt import PROMPT
from graph.retrievals.supabase_retriever import table_is_empty

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🎉 Application starting up...")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("📁 Upload directory ready")
    yield
    logger.info("🛑 Application shutting down... Bye!")

app = FastAPI(
    title="CSV/TXT QA API",
    description="Upload CSV/TXT and query via LLM chain with Supabase retrievers",
    version="1.0.0",
    lifespan=lifespan,
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    logger.info(f"📥 {request.method} {request.url.path} - start")
    response = await call_next(request)
    elapsed = time.time() - start
    logger.info(f"📤 {request.method} {request.url.path} -> {response.status_code} in {elapsed:.3f}s")
    return response

ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "https://127.0.0.1:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("./uploads")

app.state.csv_path: Optional[str] = None
app.state.txt_path: Optional[str] = None

DOCS_TABLE = "documents"

class AskIn(BaseModel):
    q: str

def collection_ready() -> bool:
    return bool(app.state.csv_path and app.state.txt_path)

def build_chain():
    if not collection_ready():
        if (table_is_empty(supabase_client, "sales_collection")
            or table_is_empty(supabase_client, "faq_collection")):
                raise HTTPException(status_code=409, detail="Collection empty. Upload a CSV and a TXT via /upload first.")
        else:
            retrievers = [
                data_retriever(supabase_client, None, None),
                all_csv_docs_retriever(supabase_client),
            ]
    else:
        retrievers = [
            data_retriever(supabase_client, app.state.csv_path, app.state.txt_path),
            all_csv_docs_retriever(supabase_client),
        ]
    return llm_chain(retrievers, PROMPT)

ALLOWED_EXTS = {".csv", ".txt"}
ALLOWED_MIME = {"text/csv", "text/plain"}

def _is_allowed(file: UploadFile) -> bool:
    name = (file.filename or "").lower()
    ext_ok = any(name.endswith(ext) for ext in ALLOWED_EXTS)
    mime_ok = (file.content_type or "").lower() in ALLOWED_MIME
    return ext_ok and mime_ok

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    violations = [
        {"filename": f.filename, "content_type": f.content_type}
        for f in files if not _is_allowed(f)
    ]
    if violations:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "message": "Only .csv and .txt are allowed",
                "allowed_extensions": list(ALLOWED_EXTS),
                "allowed_mime_types": list(ALLOWED_MIME),
                "rejected": violations,
            },
        )
    saved = []
    try:
        for f in files:
            dest = UPLOAD_DIR / f.filename
            with dest.open("wb") as out:
                for chunk in iter(lambda: f.file.read(1024 * 1024), b""):
                    out.write(chunk)
            saved.append(str(dest))
            if f.filename.lower().endswith(".csv"):
                app.state.csv_path = str(dest)
            elif f.filename.lower().endswith(".txt"):
                app.state.txt_path = str(dest)
        return {
            "uploaded": len(saved),
            "files": saved,
            "csv_path": app.state.csv_path,
            "txt_path": app.state.txt_path,
            "ready": collection_ready(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/upload")
async def upload_info():
    return {
        "detail": "Use POST /upload with multipart form-data field 'files' (CSV and TXT only)",
        "allowed_extensions": list(ALLOWED_EXTS),
        "allowed_mime_types": list(ALLOWED_MIME),
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "ready": collection_ready()}

@app.get("/")
async def root():
    return {
        "message": "Welcome to CSV/TXT QA API",
        "ready": collection_ready(),
        "endpoints": {
            "health": "/health",
            "upload": "/upload (POST, multipart form-data: files)",
            "ask": "/ask (POST, JSON: {q})",
        },
    }

@app.post("/ask")
async def ask(body: AskIn):
    chain = build_chain()
    try:
        output = chain.invoke(body.q)
        return {"query": body.q, "result": output}
    except Exception as e:
        logger.exception("Ask failed")
        raise HTTPException(status_code=500, detail=f"Ask failed: {str(e)}")

@app.get("/ask")
async def ask_info():
    return {"detail": "Use POST /ask with JSON body {\"q\": \"...\"}"}

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting CSV/TXT QA API server...")
    logger.info("🌐 http://localhost:8000  📚 /docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
