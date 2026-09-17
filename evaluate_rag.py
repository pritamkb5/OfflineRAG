import json
import ollama
from qdrant_client import QdrantClient

QDRANT_PATH = "./qdrant_storage"
COLLECTION_NAME = "pdf_documents"
EMBED_MODEL = "nomic-embed-text"
JUDGE_MODEL = "llama3"

client = QdrantClient(path=QDRANT_PATH)

def retrieve_context(query: str, top_k: int = 3):
    query_vector = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    )
    return [hit.payload["text"] for hit in response.points]

def generate_answer(query: str, context: list):
    prompt = f"Context:\n{chr(10).join(context)}\n\nQuestion: {query}\nAnswer:"
    res = ollama.chat(model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}])
    return res["message"]["content"]

def evaluate_metrics(question: str, context: list, answer: str):
    eval_prompt = f"""
You are an impartial AI evaluator assessing a RAG system.
Evaluate the following on a scale from 0.0 to 1.0:

1. Faithfulness: Are the claims in the answer strictly supported by the retrieved context? (1.0 = zero hallucinations, 0.0 = completely fabricated)
2. Answer Relevancy: Does the answer directly address the user's question? (1.0 = completely answers the query, 0.0 = completely irrelevant)
3. Context Precision: Are the retrieved context chunks relevant to answering the question? (1.0 = highly relevant, 0.0 = useless context)

Input Data:
- Question: {question}
- Retrieved Context: {chr(10).join(context)}
- Generated Answer: {answer}

Output STRICTLY valid JSON with no additional commentary:
{{
  "faithfulness": <float>,
  "answer_relevancy": <float>,
  "context_precision": <float>
}}
"""
    response = ollama.chat(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": eval_prompt}],
        options={"temperature": 0.0}
    )
    raw = response["message"]["content"].strip()
    try:
        clean_json = raw[raw.find("{"):raw.rfind("}")+1]
        return json.loads(clean_json)
    except Exception:
        return {"raw_output": raw}

if __name__ == "__main__":
    test_questions = [
        "What is the main topic covered in this document?",
        "What are the key findings or takeaways?",
    ]

    print("\n Running RAG Quality Evaluation...\n" + "="*45)
    for q in test_questions:
        ctx = retrieve_context(q)
        ans = generate_answer(q, ctx)
        scores = evaluate_metrics(q, ctx, ans)
        
        print(f"Question: {q}")
        print(f"Answer: {ans[:80]}...")
        print(f"Scores: {scores}\n" + "-"*45)