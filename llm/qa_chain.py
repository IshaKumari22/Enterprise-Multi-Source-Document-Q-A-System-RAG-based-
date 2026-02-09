from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"
)

def generate_answer(query, reranked_docs):
    if not reranked_docs:
        return "No relevant context found."

    context = "\n".join(doc.page_content for doc in reranked_docs)

    result = qa_pipeline(
        question=query,
        context=context
    )

    if result["score"] >= 0.2:
        return result["answer"]

    return (
        "Exact answer not found.\n\n"
        "Relevant information from the document:\n"
        + reranked_docs[0].page_content[:500]
    )
