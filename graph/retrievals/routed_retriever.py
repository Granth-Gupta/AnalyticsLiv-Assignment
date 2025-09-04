from typing import List, Dict, Optional
from pydantic import PrivateAttr
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

class RoutedDocsRetriever(BaseRetriever):
    _artifacts: Dict = PrivateAttr(default_factory=dict)
    _router_prompt: ChatPromptTemplate = PrivateAttr()

    def __init__(self, artifacts: Dict , prompt: ChatPromptTemplate, **data):
        super().__init__(**data)
        self._artifacts = artifacts
        self._router_prompt = prompt

    def _pick_labels(self, query: str) -> str:
        _router_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
        _labeler = self._router_prompt | _router_llm | StrOutputParser()
        try:
            return _labeler.invoke({"question": query}).strip().upper()
        except Exception:
            return "SALES_COMPLETE+FAQ"

    def _resolve_retrievers(self, label: str) -> List[Optional[BaseRetriever]]:
        faq = self._artifacts.get("retrieval_faq")
        sample = self._artifacts.get("retrieval_sales")
        complete = self._artifacts.get("all_sales_retriever")

        if label == "FAQ":
            print("Retrieving FAQ")
            return [faq]
        if label == "SALES_SAMPLE":
            print("Retrieving SALES SAMPLE")
            return [sample]
        if label == "SALES_COMPLETE":
            print("Retrieving SALES COMPLETE")
            return [complete]
        if label == "SALES_COMPLETE+FAQ":
            print("Retrieving FAQ+SALES COMPLETE")
            return [complete, faq]
        if label == "SALES_SAMPLE+FAQ":
            print("Retrieving SALES SAMPLE + FAQ")
            return [sample, faq]
        # Conservative default
        print("Retrieving All retrievers")
        return [complete, sample, faq]

    def pick_retrievers(self, query: str):
        label = self._pick_labels(query)
        return [r for r in self._resolve_retrievers(label) if r is not None]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        label = self._pick_labels(query)
        chosen = [r for r in self._resolve_retrievers(label) if r is not None]
        if not chosen:
            raise RuntimeError("No retrievers configured in artifacts. Expected keys: "
                               "'retrieval_faq', 'retrieval_sales', 'retrieval_all_sales'.")

        docs: List[Document] = []
        for retr in chosen:
            if hasattr(retr, "invoke"):
                docs.extend(retr.invoke(query, config={"callbacks": run_manager.get_child()}))
            else:
                docs.extend(retr.get_relevant_documents(query, callbacks=run_manager.get_child()))
        return docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)
