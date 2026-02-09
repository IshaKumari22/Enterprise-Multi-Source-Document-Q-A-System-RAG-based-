#  Enterprise Multi-Source RAG System (Free & Open-Source)

An end-to-end **Retrieval-Augmented Generation (RAG)** application that allows users to upload documents (PDF/CSV/Web URLs) and ask natural-language questions.  
The system retrieves **relevant chunks**, reranks them, and extracts **accurate answers strictly from the uploaded documents** using open-source models.

---

##  Features

- Upload **PDF / CSV documents**
- Automatic **text extraction**
- Smart **chunking**
- **Vector search** using FAISS
- **Semantic retrieval**
- **Cross-encoder reranking**
- **Extractive Question Answering (QA)**  
- Works with **any document type** (HR policy, research paper, resume, reports, manuals, etc.)
- **100% free & open-source** (No OpenAI API)

---

##  Architecture (RAG Pipeline)

User Uploads Documents
↓
Text Extraction (pdfplumber / CSV loader)
↓
Chunking
↓
Embeddings (Sentence Transformers)
↓
Vector Store (FAISS)
↓
Semantic Retrieval
↓
Cross-Encoder Reranking
↓
Extractive QA Model
↓
Final Answer (From document only)

##  Tech Stack

### Core
- **Python**
- **Streamlit** – UI
- **FAISS** – Vector database

### NLP / ML
- **Sentence-Transformers** (`all-MiniLM-L6-v2`)
- **Hugging Face Transformers**
- **deepset/roberta-base-squad2** (Extractive QA)
- **Cross-Encoder reranker**

### Document Processing
- **pdfplumber**
- **spaCy**


# Author

Built by Isha
Aspiring AI / Full-Stack Engineer