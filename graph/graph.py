from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from retrievals.data_retriever import data_retriever

PERSIST_DIR = "../ChromaDB"
CSV_PATH = "./../assets/sales_data.csv"
TXT_PATH = "./../assets/faq_data.txt"

load_dotenv()

template = """Answer the question based only on the following context:
{sales_csv_context}
{FAQ_context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# model = OllamaLLM(model="llama3.2:latest")
model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature=0.)
sales_retriever, faq_retriever = data_retriever(PERSIST_DIR, CSV_PATH, TXT_PATH)

chain = (
    {
        "sales_csv_context": sales_retriever,
        "FAQ_context": faq_retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)

# print(chain.invoke("Show me the average sales along with refund policy."))
print(chain.invoke("Provide list of total sales of 2023 for all months from Jan to Dec."))