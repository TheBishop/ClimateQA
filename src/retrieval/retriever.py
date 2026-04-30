from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"


def load_retriever(k: int = 5):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",           # max marginal relevance — avoids duplicate chunks
        search_kwargs={"k": k, "fetch_k": k * 3},
    )
    return retriever


def retrieve(query: str, k: int = 5) -> list:
    retriever = load_retriever(k=k)
    docs = retriever.invoke(query)
    return docs