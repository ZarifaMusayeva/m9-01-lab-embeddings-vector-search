import os
import json
import numpy as np
import google.generativeai as genai

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Error: GOOGLE_API_KEY not found in environment variables.")

genai.configure(api_key=api_key)
MODEL_NAME = "models/gemini-embedding-001"

def get_embedding(text: str) -> np.ndarray:
    response = genai.embed_content(
        model=MODEL_NAME,
        content=text,
        task_type="retrieval_document"
    )
    return np.array(response['embedding'])

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

print("Loading knowledge base...")
with open("knowledge_base.json", "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

print("Generating embeddings for passages...")
embedded_passages = []
for item in knowledge_base:
    text_content = item.get("text") or item.get("body") or ""
    if not text_content:
        continue
    
    vector = get_embedding(text_content)
    embedded_passages.append({
        "id": item.get("id"),
        "source": item.get("source", "unknown"),
        "text": text_content,
        "vector": vector
    })
print("Knowledge base embeddings ready.\n")

queries = [
    "my laptop won't switch on",
    "how do I stop being billed every month?",
    "access denied error when saving a file",
    "where do I leave my car in the evening?",
    "what's the wifi password?"
]

print("Starting Semantic Search Results:\n" + "="*60)

for query in queries:
    print(f"\nQuery: \"{query}\"")
    
    query_vector = get_embedding(query)
    
    scores = []
    for item in embedded_passages:
        score = cosine_similarity(query_vector, item["vector"])
        scores.append((score, item))
    
    scores.sort(key=lambda x: x[0], reverse=True)
    
    print("Top 3 Matches:")
    for rank, (score, item) in enumerate(scores[:3], 1):
        print(f"  {rank}. [Score: {score:.4f}] (Source: {item['source']})")
        print(f"     Text: {item['text']}")
    print("-" * 60)



# ============================================================
# REFLECTION
# ============================================================
# Query 1: "my laptop won't switch on"
#   Best match shares NO key words with the query.
#   "switch on" matched "won't turn on" — the model captured
#   the concept of a device failing to start, not the words.
#
# Query 2: "how do I stop being billed every month?"
#   NO word overlap. "billed" / "every month" matched
#   "subscription" / "billing period" — pure semantic match.
#
# Query 3: "access denied error when saving a file"
#   PARTIAL overlap — "access denied" appears in the passage,
#   but "saving a file" ≠ "write permission to target folder".
#
# Query 4: "where do I leave my car in the evening?"
#   NO key word overlap. "car" → "park", "evening" → "after 6pm"
#   The embedding understood the situation, not the words.
#
# Conclusion: embeddings retrieve by MEANING. A keyword search
# would have failed on 3 of 4 queries entirely.
#
# STRETCH — "what's the wifi password?" (top score: 0.8064)
#   No passage covers wifi. Yet the score is 0.80 — dangerously
#   close to real matches. A threshold (e.g. score < 0.75 →
#   "No answer found") would prevent misleading results.
# ============================================================