import os
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

# ENVIRONMENT VARIABLES (Use dotenv or os.environ)
PINECONE_API_KEY = 'pcsk_53KbzW_9Qc9oTo42j16gSVRoghPig376R4oKMnkafv8MatjWeCCZhouKEioasKejD1zv5b'
GOOGLE_API_KEY = 'REMOVED'

pc = Pinecone(api_key=PINECONE_API_KEY)



# Your indexes
indexes = [
    "guest-messaging-guide-doc-embeddings",
    "house-notes-doc-embeddings"
]

# Gemini setup
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Embed function using Gemini's embedding model
def embed_text(text):
    embedding_model = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return embedding_model['embedding']

# Function to query Pinecone and generate response using Gemini
def answer_query(query, top_k=10):
    query_vector = embed_text(query)
    all_results = []

    for index_name in indexes:
        index = pc.Index(index_name)  # ✅ fixed here
        result = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        matches = result.get('matches', [])
        all_results.extend(matches)

    # Sort by score and limit to top_k
    top_matches = sorted(all_results, key=lambda x: x['score'], reverse=True)[:top_k]
    context = "\n\n".join([match['metadata'].get('text', '') for match in top_matches])

    prompt = f"""Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:"""

    response = gemini_model.generate_content(prompt)
    return response.text



# Example usage
if __name__ == "__main__":
    user_question = input("Ask your question: ")
    answer = answer_query(user_question)
    print("\n--- Answer ---")
    print(answer)
