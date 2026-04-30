import os
import glob
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

CHROMA_PATH = "chroma_db"
DATA_PATH = "data/pdfs"
EMBED_MODEL = "all-MiniLM-L6-v2"  # fast, 80MB, good quality


def load_pdfs(data_path: str) -> list:
    docs = []
    pdf_files = glob.glob(os.path.join(data_path, "*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {data_path}/")
        return docs
    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        docs.extend(loader.load())
    print(f"Loaded {len(docs)} pages from {len(pdf_files)} PDF(s)")
    return docs


def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks: list):
    print(f"Loading embedding model: {EMBED_MODEL} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
    )
    print("Building ChromaDB vectorstore ...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )
    print(f"Vectorstore saved to '{CHROMA_PATH}/' with {len(chunks)} chunks")
    return vectorstore


def main():
    docs = load_pdfs(DATA_PATH)
    if not docs:
        print("Add PDFs to data/pdfs/ and re-run.")
        return
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)
    print("Ingestion complete!")


if __name__ == "__main__":
    main()