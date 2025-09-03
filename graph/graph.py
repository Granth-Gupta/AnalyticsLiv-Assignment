from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from typing import List, Tuple

load_dotenv()

def llm_chain(retrievers: List, prompt: str):

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.)

    sales_retriever, faq_retriever, vectorstore_csv, _ = retrievers[0]
    all_sales_retriever = retrievers[1]

    prompt = ChatPromptTemplate.from_template(prompt)

    chain = (
            {
                "sales_csv_context": sales_retriever,
                "all_sales_context": all_sales_retriever,
                "FAQ_context": faq_retriever,
                "question": RunnablePassthrough()
            }
            | prompt
            | model
            | StrOutputParser()
    )

    return chain