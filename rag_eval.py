import json
import numpy as np
import chromadb
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---------------------------
# Load knowledge base 
# ---------------------------
with open("knowledge_base.json", "r") as f:
    kb = json.load(f)

texts = [doc["text"] for doc in kb]
ids = [doc["id"] for doc in kb]

id_to_text = {doc["id"]: doc["text"] for doc in kb}

# ---------------------------
# Models 
# ---------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------------------------
# ChromaDB setup 
# ---------------------------
client = chromadb.Client()

collection = client.get_or_create_collection("rag_knowledge_base")

embeddings = embed_model.encode(texts).tolist()

collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=ids
)

# ---------------------------
# Eval set
# ---------------------------
eval_set = [
    {"q": "Where can employees park after 6pm?", "expected": "kb-01"},
    {"q": "How do I turn on a device that won't start?", "expected": "kb-02"},
    {"q": "What is the annual leave policy?", "expected": "kb-03"},
    {"q": "What is the refund policy after purchase?", "expected": "kb-04"},
    {"q": "What does error code 0x80070005 mean?", "expected": "kb-08"},
]

# ---------------------------
# Baseline retrieval (dense)
# ---------------------------
def retrieve_baseline(query, k=3):
    q_emb = embed_model.encode([query])[0]
    results = collection.query(query_embeddings=[q_emb], n_results=k)
    return results["ids"][0]

# ---------------------------
# Rerank retrieval
# ---------------------------
def retrieve_rerank(query, k=3, pool_size=8):
    q_emb = embed_model.encode([query])[0]

    results = collection.query(query_embeddings=[q_emb], n_results=pool_size)

    docs = results["documents"][0]
    doc_ids = results["ids"][0]

    pairs = [(query, doc) for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)

    return [x[0] for x in ranked[:k]]

# ---------------------------
# Hit rate
# ---------------------------
def hit_rate(retriever):
    hits = 0

    for item in eval_set:
        retrieved = retriever(item["q"])
        if item["expected"] in retrieved:
            hits += 1

    return hits / len(eval_set)

# ---------------------------
# Ollama faithfulness judge
# ---------------------------
def judge_faithfulness(question, context, answer):
    prompt = f"""
You are a strict evaluator.

Question: {question}

Context:
{context}

Answer:
{answer}

Is the answer fully supported by the context? Reply only YES or NO.
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    result = response.json()["response"]
    return "YES" in result.upper()

# --------------------------- 
# Faithfulness evaluation 
# --------------------------- 
def evaluate_faithfulness(retriever):
    results = []

    for item in eval_set:
        retrieved_ids = retriever(item["q"])

        context = "\n".join([id_to_text[i] for i in retrieved_ids])

        answer = context  # simple RAG baseline answer

        ok = judge_faithfulness(item["q"], context, answer)
        results.append(ok)

    return sum(results) / len(results)

# ---------------------------
# Run evaluation
# ---------------------------
print("\n=== RETRIEVAL RESULTS ===")
print("Baseline hit rate:", hit_rate(retrieve_baseline))
print("Rerank hit rate:", hit_rate(retrieve_rerank))

print("\n=== FAITHFULNESS RESULTS ===")
print("Baseline faithfulness:", evaluate_faithfulness(retrieve_baseline))
print("Rerank faithfulness:", evaluate_faithfulness(retrieve_rerank)) 