from langchain_core.prompts import ChatPromptTemplate

ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """
    You are an agent to find the required labels

    Labels:
    - SALES_COMPLETE
    - SALES_SAMPLE
    - SALES_SAMPLE+FAQ
    - SALES_COMPLETE+FAQ

    - We are working on sales data (with fields like "total sales", "year", "month", etc.) and faq data (contain policies/FAQ in form of question(Q) and answers (A)
    - If query is global like , "find all total sales in 2024" return 
            SALES_COMPLETE
    - If query like "find month in which total sales is 1000", return 
            SALES_SAMPLE
    - If query is like "find month in which total sales is 1000 and provide refund policy", return 
            SALES_SAMPLE+FAQ
    - If query is like "Find average total sales in 2024 i.e. global coverage and return policies then it means it require both SALES_COMPLETE and FAQ thus return 
            SALES_COMPLETE+FAQ
    - i.e. always return one label
    Classify the question as one of the following labels, or a plus-combination if both are required:
    - FAQ, SALES_SAMPLE, SALES_COMPLETE, SALES_COMPLETE+FAQ, SALES_SAMPLE+FAQ
    Question: {question}
    Answer with one label exactly.
    """
)