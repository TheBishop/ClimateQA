import streamlit as st
# st.write("Key preview:", st.secrets.get("GROQ_API_KEY", "NOT FOUND")[:8])
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.api.chain import ask
from src.ingestion.ingest import main as run_ingest

st.set_page_config(
    page_title="ClimateQA",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 ClimateQA")
st.caption("Ask questions about climate science — powered by IPCC reports & Groq LLaMA 3")

# Sidebar: ingestion controls
with st.sidebar:
    st.header("📂 Document Index")
    chroma_exists = os.path.exists("chroma_db")
    if chroma_exists:
        st.success("✅ Index ready")
    else:
        st.warning("⚠️ No index found. Add PDFs to data/pdfs/ then click below.")

    if st.button("🔄 (Re)build Index", use_container_width=True):
        with st.spinner("Ingesting PDFs and building vectorstore..."):
            run_ingest()
        st.success("Index built!")
        st.rerun()

    st.divider()
    st.markdown("**Stack**")
    st.markdown("- 🤖 LLM: LLaMA 3.3 70B (Groq)\n- 🔍 Embeddings: all-MiniLM-L6-v2\n- 🗄️ VectorDB: ChromaDB")

# Main chat area
if not chroma_exists:
    st.info("👈 Add PDFs to `data/pdfs/` and build the index using the sidebar.")
    st.stop()

# Sample questions
st.markdown("**Try a question:**")
cols = st.columns(3)
examples = [
    "What are the projected temperature increases by 2100?",
    "How does climate change affect sea level rise?",
    "What are the main drivers of global warming?",
]
for i, example in enumerate(examples):
    if cols[i].button(example, use_container_width=True):
        st.session_state["question"] = example

# Query input
question = st.text_input(
    "Ask a climate question:",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What does the IPCC say about Arctic ice loss?",
)

if st.button("🔍 Ask", type="primary", use_container_width=True) and question:
    with st.spinner("Retrieving and generating answer..."):
        result = ask(question)

    st.markdown("### Answer")
    st.markdown(result["answer"])

    st.markdown("### 📎 Sources Retrieved")
    for i, src in enumerate(result["sources"]):
        with st.expander(f"Source {i+1} — {src['file']}, page {src['page']}"):
            st.caption(src["snippet"])