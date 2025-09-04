from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def _qa_to_context(qa_chain):
    def _run(q: str) -> str:
        try:
            ans = qa_chain.invoke({"query": q})
        except Exception:
            ans = qa_chain.invoke(q)
        return ans["result"] if isinstance(ans, dict) and "result" in ans else str(ans)
    return _run

def llm_chain_faiss(artifacts: dict, prompt: str):
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

    retrievalQA_sales = artifacts["retrievalQA_sales"]
    retrievalQA_faq = artifacts["retrievalQA_faq"]

    # Wrap QA chains as Runnables that map input question -> context strings
    sales_context = _qa_to_context(retrievalQA_sales)
    faq_context = _qa_to_context(retrievalQA_faq)

    all_sales_retriever = artifacts.get("all_sales_retriever", sales_context)

    chat_prompt = ChatPromptTemplate.from_template(prompt)

    chain = (
        {
            "sales_csv_context": sales_context,   # calls RetrievalQA with the question, returns context-like text
            "all_sales_context": all_sales_retriever,
            "FAQ_context": faq_context,
            "question": RunnablePassthrough(),
        }
        | chat_prompt
        | model
        | StrOutputParser()
    )
    return chain
