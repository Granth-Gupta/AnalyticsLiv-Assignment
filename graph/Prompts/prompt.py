PROMPT = """
System task: Answer strictly from the provided contexts. Do not use outside knowledge.

Contexts:

Sales data (sample subset):
{sales_csv_context}

Sales data (complete):
{all_sales_context}

Policy / FAQ:
{FAQ_context}

Instructions:

Choose the minimal context that guarantees correctness:
- Use complete Sales data when the task needs global coverage (e.g., averages, maxima, full-year/month listings, year filters). [Must use (2)].
- Use sample Sales data only when a small subset suffices (e.g., exact match lookup like “Total_Sales == 1805”). [Prefer (1), fall back to (2) only if needed].
- Use Policy/FAQ for policy questions. [(3)]

- If multiple contexts are needed, combine at most two and explain the choice in one short clause.
- If required data is missing from all provided contexts, respond: “Insufficient data in provided context.”
- Output must be concise and to the point. No extra commentary.

Answer format:
- For sales aggregates or listings: show Month-Year and Total_Sales per line; include the calculation step/result.
- For policies: quote the relevant policy line(s) or paraphrase precisely.
- If filters are applied (e.g., Year == 2024), state the filter in one short clause.

Question:
{question}
"""