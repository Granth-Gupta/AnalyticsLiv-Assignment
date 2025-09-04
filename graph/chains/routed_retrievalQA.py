from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers import MergerRetriever
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

def retrieval_qa_chain(prompt: ChatPromptTemplate, router):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

    def invoke(query: str):
        chosen = router.pick_retrievers(query)
        if not chosen:
            raise RuntimeError("No retrievers available for the query. Check artifacts.")
        retr = chosen if len(chosen) == 1 else MergerRetriever(retrievers=chosen)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retr,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False,
            verbose=False,
        )
        return qa.invoke({"query": query})

    class OnDemandQA:
        def invoke(self, x):
            # x may be a dict or a str depending on caller
            query = x if isinstance(x, str) else x.get("query", x)
            return invoke(query)

    return OnDemandQA()

