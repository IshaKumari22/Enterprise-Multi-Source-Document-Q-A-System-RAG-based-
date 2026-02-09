import streamlit as st
from llm.qa_chain import generate_answer

from ingestion.loaders import load_all_documents
from processing.chunking import chunk_documents
from indexing.vector_index import create_faiss_index
from retrievers.semantic_retriever import retrieve_chunks
from reranker.cross_encoder_reranker import CrossEncoderReranker

if "chunks" not in st.session_state:
    st.session_state["chunks"] = []

if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

if "reranker" not in st.session_state:
    st.session_state["reranker"] = CrossEncoderReranker()



st.set_page_config(page_title="Enterprise Multi-Source RAG", layout="wide")
st.title(" Enterprise Multi-Source RAG")
st.divider()


if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


uploaded_files = st.file_uploader(
    "Upload PDF or CSV files",
    type=["pdf", "csv"],
    accept_multiple_files=True
)

website_url = st.text_input("Enter Website URL (optional)")

st.divider()

if st.button(" Create Chunks"):
    documents = load_all_documents(uploaded_files, website_url)
    st.session_state.chunks = chunk_documents(documents)

    st.success(f" Documents Loaded: {len(documents)}")
    st.success(f" Chunks Created: {len(st.session_state.chunks)}")

if st.session_state.chunks:
    st.subheader(" View Chunks")

    chunk_index = st.selectbox(
        "Select chunk number",
        range(len(st.session_state.chunks)),
        key="chunk_selector"
    )

    selected_chunk = st.session_state.chunks[chunk_index]

    st.markdown("###  Chunk Text")
    st.write(selected_chunk.page_content)

    st.markdown("###  Chunk Metadata")
    st.json(selected_chunk.metadata)

    st.markdown("###  Chunk Length")
    st.write(len(selected_chunk.page_content), "characters")


st.divider()


st.divider()

if st.session_state.chunks:
    if st.button(" Create Vector Index"):
        st.session_state.vector_store = create_faiss_index(
            st.session_state.chunks
        )
        st.success(" Vector index created using FAISS")

if st.session_state.vector_store is not None:
    st.info(" Vector index is ready")

    st.write(
        "Total vectors in index:",
        len(st.session_state.vector_store.index_to_docstore_id)
    )

if st.session_state.vector_store is not None:
    st.subheader(" Ask a Question")

    user_query = st.text_input("Enter your question")
    top_k = st.slider("Number of chunks to retrieve", 1, 5, 3)

    if user_query:
        retrieved_docs = retrieve_chunks(
            st.session_state.vector_store,
            user_query,
            k=top_k
        )

        reranked_docs = st.session_state["reranker"].rerank(
            user_query,
            retrieved_docs,
            top_k=top_k
        )

        st.success(f"Retrieved & reranked {len(reranked_docs)} chunks")

        for i, doc in enumerate(reranked_docs):
            st.markdown(f"###  Reranked Chunk {i+1}")
            st.write(doc.page_content)

        st.divider()
        st.subheader(" Final Answer")

        final_answer = generate_answer(
            user_query,
            reranked_docs
        )

        st.subheader(" Final Answer")
        st.success(final_answer)


