from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_template("""
System task: Answer strictly from the provided context. Do not use outside knowledge. If the needed evidence is not present, respond: “Insufficient data in provided context.”

Context:
{context}

Instructions:
- Use complete Sales for global coverage (aggregates/max/min/year filters/listings).
- Prefer sample Sales only for exact matches; otherwise use complete Sales.
- Use Policy/FAQ for policy or procedural questions.
- If combining sources, use at most two and justify briefly.

Answer format:
- For sales aggregates/listings: Provide small and concise result.
- For policies: quote or precisely paraphrase relevant lines.
- If filters are applied (e.g., Year == 2024), state the filter briefly.
- Answer should only contain information required i.e. do not unnecessarily increase length of the answer.

Question:
{question}
""")