from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents, top_k=3):
        """
        Re-rank retrieved documents using a cross-encoder
        """
        pairs = [(query, doc.page_content) for doc in documents]

        scores = self.model.predict(pairs)

        scored_docs = list(zip(documents, scores))

        scored_docs.sort(key=lambda x: x[1], reverse=True)

        reranked_docs = [doc for doc, score in scored_docs[:top_k]]
        return reranked_docs
