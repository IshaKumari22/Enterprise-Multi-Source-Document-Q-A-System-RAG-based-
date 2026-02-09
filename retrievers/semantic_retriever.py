def retrieve_chunks(vector_store, query, k=3):
    """
    Retrieve top-k relevant chunks for a query
    """
    results = vector_store.similarity_search(query, k=k)
    return results
