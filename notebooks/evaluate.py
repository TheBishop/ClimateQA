import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from src.api.chain import build_chain
from src.retrieval.retriever import load_retriever

TEST_QUESTIONS = [
    "What is the observed increase in global surface temperature since pre-industrial times?",
    "What are the projected temperature increases under SSP5-8.5 by 2100?",
    "How much has sea level risen since 1900?",
    "What is the likely range of global warming under SSP1-1.9?",
    "How confident is the IPCC that human influence has warmed the climate?",
    "What role does methane play in climate change according to AR6?",
    "What happens to Arctic sea ice under different warming scenarios?",
    "How does climate change affect extreme precipitation events?",
    "What is the carbon budget remaining for 1.5 degrees of warming?",
    "How have greenhouse gas concentrations changed since pre-industrial times?",
    "What is the projected change in global mean sea level by 2100 under SSP5-8.5?",
    "How does the rate of sea level rise compare to previous centuries?",
    "What are the consequences of exceeding 2 degrees of global warming?",
    "How has the frequency of hot extremes changed since the 1950s?",
    "What does the IPCC say about irreversible climate changes?",
    "How does soil moisture change under future warming scenarios?",
    "What is the relationship between cumulative CO2 emissions and warming?",
    "How confident is the IPCC about future precipitation changes?",
    "What does AR6 say about the Atlantic Meridional Overturning Circulation?",
    "How have ocean heat content and acidification changed?",
]

def run_evaluation():
    groq_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    # Inject Groq into RAGAS metrics
    faithfulness.llm = groq_llm
    answer_relevancy.llm = groq_llm
    answer_relevancy.embeddings = embeddings

    print("Building chain and retriever...")
    chain, retriever = build_chain()

    questions, answers, contexts = [], [], []

    for i, question in enumerate(TEST_QUESTIONS):
        print(f"[{i+1}/{len(TEST_QUESTIONS)}] {question[:60]}...")
        docs = retriever.invoke(question)
        answer = chain.invoke(question)
        questions.append(question)
        answers.append(answer)
        contexts.append([d.page_content for d in docs])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    })

    print("\nRunning RAGAS evaluation (this takes a few minutes)...")

    import time
    time.sleep(10)  # let token bucket refill before evaluation
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=groq_llm,
        embeddings=embeddings,
    )

    print("\n" + "="*50)
    print("RAGAS EVALUATION RESULTS")
    print("="*50)
    df = results.to_pandas()
    print(f"Faithfulness:      {df['faithfulness'].mean():.3f}")
    print(f"Answer Relevancy:  {df['answer_relevancy'].mean():.3f}")
    print("="*50)

    df.to_csv("notebooks/ragas_results.csv", index=False)
    print("Full results saved to notebooks/ragas_results.csv")

if __name__ == "__main__":
    run_evaluation()
