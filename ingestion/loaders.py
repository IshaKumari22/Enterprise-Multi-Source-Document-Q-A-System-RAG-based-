import tempfile
import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    WebBaseLoader
)


def load_pdfs(uploaded_files):
    documents = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = "pdf"
            doc.metadata["filename"] = uploaded_file.name

        documents.extend(docs)

        os.remove(temp_path)

    return documents


def load_csvs(uploaded_files):
    documents = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        loader = CSVLoader(file_path=temp_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = "csv"
            doc.metadata["filename"] = uploaded_file.name

        documents.extend(docs)

        os.remove(temp_path)

    return documents


def load_website(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = "website"
        doc.metadata["url"] = url

    return docs


def load_all_documents(uploaded_files, website_url):
    all_documents = []

    pdf_files = [f for f in uploaded_files if f.name.endswith(".pdf")]
    csv_files = [f for f in uploaded_files if f.name.endswith(".csv")]

    if pdf_files:
        all_documents.extend(load_pdfs(pdf_files))

    if csv_files:
        all_documents.extend(load_csvs(csv_files))

    if website_url:
        all_documents.extend(load_website(website_url))

    return all_documents
