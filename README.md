
# 🌍 ClimateQA

A retrieval-augmented generation (RAG) system for querying IPCC climate reports using natural language.

[**Live Demo →**](https://climateapp.streamlit.app)

![ClimateQA screenshot](docs/screenshot.png)

## The Problem
Climate scientists and policymakers need fast, accurate access to IPCC findings. Reading 3,000+ pages of AR6 reports manually is impractical. ClimateQA lets you ask plain-language questions and get cited, grounded answers in seconds.

## Architecture

IPCC AR6 PDFs
↓
[Ingest] PyPDF → RecursiveCharacterTextSplitter (800 tokens, 150 overlap)
↓
[Embed] all-MiniLM-L6-v2 (local, CPU) → ChromaDB (MMR retrieval, k=5)
↓
[Generate] LLaMA 3.3 70B via Groq → answer + page-level citations
↓
[UI] Streamlit


## Evaluation (RAGAS, n=20 questions)

| Metric | Value |
|--------|-------|
| Faithfulness | 0.714 |
| Answer Relevancy | 0.973 |

Evaluated on 20 domain-specific questions drawn from IPCC AR6 WG1 topics.

## Tech Stack

| Layer | Tool |
|-------|------|
| LLM | LLaMA 3.3 70B (Groq) |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Vector DB | ChromaDB (MMR retrieval) |
| Orchestration | LangChain |
| UI | Streamlit |

## Quick Start

```bash
git clone https://github.com/TheBishop/ClimateQA
cd ClimateQA
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your GROQ_API_KEY
python -m src.ingestion.ingest
streamlit run app.py
```

## Project Structure

ClimateQA/
├── app.py                  # Streamlit UI
├── src/
│   ├── ingestion/ingest.py # PDF loading, chunking, embedding
│   ├── retrieval/retriever.py # ChromaDB MMR retrieval
│   └── api/chain.py        # RAG chain (LangChain + Groq)
├── notebooks/
│   ├── evaluate.py         # RAGAS evaluation script
│   └── ragas_results.csv   # Full evaluation output
└── data/pdfs/              # Source documents

## Author

**Dzahene Richmond Elorm** — Teaching & Research Assistant, Dept. of Meteorology & Climate Science, KNUST  
[github.com/TheBishop](https://github.com/TheBishop)









## API

```bash
curl -X POST https://your-api-url/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are projected sea level rises by 2100?"}'
```

Interactive docs available at `/docs` (Swagger UI).
