import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.retrieval.retriever import load_retriever

load_dotenv()

SYSTEM_PROMPT = """You are ClimateQA, an expert assistant that answers questions about \
climate science using information from IPCC reports and authoritative climate documents.

Answer ONLY based on the context below. If the context doesn't contain enough information \
to answer confidently, say so clearly — do not fabricate facts.

For each key claim, indicate which source document it comes from (use the source filename).

Context:
{context}
"""

def get_api_key():
    key = None
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        # Covers ImportError (streamlit not installed) and
        # StreamlitSecretNotFoundError (no secrets.toml present)
        key = None
    return key or os.getenv("GROQ_API_KEY")


def get_langfuse_handler():
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return None
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        return None
    return CallbackHandler(
        public_key=public_key,
        secret_key=secret_key,
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )


def format_docs(docs):
    parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Source {i+1}: {os.path.basename(source)}, p.{page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_chain():
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Set it via streamlit secrets or the GROQ_API_KEY environment variable."
        )
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        api_key=api_key,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    return (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def ask(question: str) -> dict:
    retriever = load_retriever(k=5)
    docs = retriever.invoke(question)

    if not docs:
        return {
            "answer": "No relevant documents were found for that question.",
            "sources": [],
        }

    langfuse_handler = get_langfuse_handler()
    config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}

    chain = build_chain()
    answer = chain.invoke(
        {"context": format_docs(docs), "question": question},
        config=config,
    )

    sources = [
        {
            "file": os.path.basename(d.metadata.get("source", "unknown")),
            "page": d.metadata.get("page", "?"),
            "snippet": d.page_content[:200] + "...",
        }
        for d in docs
    ]
    return {"answer": answer, "sources": sources}
