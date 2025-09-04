from typing import List
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr

SALES_TABLE = "sales_collection"

class AllCSVRetriever(BaseRetriever):
    _supabase_client: any = PrivateAttr(default=None)

    def __init__(self, supabase_client, **data):
        super().__init__(**data)
        self._supabase_client = supabase_client

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        client = self._supabase_client
        page_size = 100
        docs_all: List[Document] = []
        start = 0
        while True:
            end = start + page_size - 1
            resp = client.table(SALES_TABLE).select("content, metadata").range(start, end).execute()
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
        return docs_all
