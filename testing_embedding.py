import os
import re
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Initialize Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Pinecone index
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("house-information-embeddings")

# --- Embedding helper ---
def embed_query(text: str) -> List[float]:
    if not text.strip():
        return []
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_query"
    )
    return response['embedding']


# --- Reranking helper ---
def rerank_passages(passages: List[str], query: str) -> List[str]:
    query_words = set(query.lower().split())
    ranked = sorted(passages, key=lambda p: sum(1 for w in query_words if w in p.lower()), reverse=True)
    return ranked


# --- Deduplicate helper ---
def deduplicate(passages: List[str]) -> List[str]:
    seen = set()
    unique = []
    for text in passages:
        norm = re.sub(r'\s+', ' ', text.lower().strip())
        if norm not in seen:
            seen.add(norm)
            unique.append(text)
    return unique


# --- Build context ---
def build_context(passages: List[str], max_len: int = 6000) -> str:
    deduped = deduplicate(passages)
    context = " ".join(deduped)
    return context[:max_len]


# --- Generate answer ---
def generate_response(context: str, question: str) -> str:
    prompt = f"""You are a real estate assistant AI. Use the following internal house notes to answer the user's question.

House Information:
{context}

User Question:
{question}

Give a concise, factual response based on the house notes, filling gaps if needed but never hallucinating beyond context.
"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                top_p=0.9,
                top_k=40,
                max_output_tokens=1200
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        return "Sorry, I couldn't generate a response."


# --- Main chat function ---
def answer_house_question(query: str, top_k: int = 20) -> str:
    query_emb = embed_query(query)
    if not query_emb:
        return "Invalid query."

    results = index.query(
        vector=query_emb,
        top_k=top_k,
        include_metadata=True
    )

    matches = [m["metadata"].get("text", "") for m in results["matches"] if "metadata" in m]
    if not matches:
        return "No relevant house information found."

    reranked = rerank_passages(matches, query)
    context = build_context(reranked)
    return generate_response(context, query)


# --- Example usage ---
if __name__ == "__main__":
    question = "tell me about the 140 Navajo Dr"
    answer = answer_house_question(question)
    print(f"\nAnswer: {answer}")
