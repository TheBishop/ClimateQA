from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

ROOT = Path(__file__).resolve().parents[2]
CHROMA_PATH = ROOT / "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"


def load_retriever(k: int = 5):
    if not CHROMA_PATH.exists():
        raise FileNotFoundError(
            f"Chroma vectorstore not found at '{CHROMA_PATH}'. "
            "First run ingestion by adding PDFs to data/pdfs and executing src/ingestion/ingest.py."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
    )
    vectorstore = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",          
        search_kwargs={"k": k, "fetch_k": k * 3},
    )
    return retriever


def retrieve(query: str, k: int = 5) -> list:
    retriever = load_retriever(k=k)
    docs = retriever.invoke(query)
    return docs