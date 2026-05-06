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

# def get_api_key():
#     try:
#         import streamlit as st
#         return st.secrets["GROQ_API_KEY"]
#     except Exception:
#         return os.getenv("GROQ_API_KEY")

def get_api_key():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY is not set")
    return key

def format_docs(docs):
    parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Source {i+1}: {os.path.basename(source)}, p.{page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)

def build_chain():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        api_key=get_api_key(),
    )
    retriever = load_retriever(k=5)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

def ask(question: str) -> dict:
    chain, retriever = build_chain()
    docs = retriever.invoke(question)
    answer = chain.invoke(question)
    sources = [
        {
            "file": os.path.basename(d.metadata.get("source", "unknown")),
            "page": d.metadata.get("page", "?"),
            "snippet": d.page_content[:200] + "...",
        }
        for d in docs
    ]
    return {"answer": answer, "sources": sources}
