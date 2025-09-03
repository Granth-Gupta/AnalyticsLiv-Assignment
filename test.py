from Supabase.client import supabase_client
from graph.graph import llm_chain
from graph.retrievals.supabase_retriever import data_retriever, all_csv_docs_retriever
from graph.Prompts.prompt import PROMPT

CSV_PATH = "assets/sales_data.csv"
TXT_PATH = "assets/faq_data.txt"

retrievers = [data_retriever(supabase_client, CSV_PATH, TXT_PATH), all_csv_docs_retriever(supabase_client)]

if __name__ == '__main__':
    chain = llm_chain(retrievers, PROMPT)

    user_query = input("Enter user query: \n")
    result = chain.invoke(user_query)
    print("-----------------------------------\n")
    print("User Query: ", user_query)
    print("Result: ", result)