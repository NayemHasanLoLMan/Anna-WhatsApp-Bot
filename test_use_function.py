import os
import re
import openai
import numpy as np
from typing import Dict, Optional, List
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

class PineconeRAGQuery:
    def __init__(self, openai_api_key: str, pinecone_api_key: str, pinecone_index_name: str):
        # Initialize OpenAI
        openai.api_key = openai_api_key
        self.embedding_model = "text-embedding-ada-002"  # Updated to newer model
        self.chat_model = "gpt-4-turbo"  # Updated to newer model
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        # Connect to existing index
        try:
            self.index = self.pc.Index(pinecone_index_name)
            print(f"✅ Connected to Pinecone index: {pinecone_index_name}")
        except Exception as e:
            print(f"❌ Error connecting to index: {e}")
            raise

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for the query using OpenAI's embedding model."""
        try:
            response = openai.Embedding.create(
                model=self.embedding_model,
                input=query
            )
            return response['data'][0]['embedding']
        except Exception as e:
            print(f"❌ Error getting embedding: {e}")
            raise

    def _query_pinecone(self, query_embedding: List[float], top_k: int = 20, house_name: Optional[str] = None):
        """Query Pinecone index with optional filtering."""
        filter_dict = {}
        if house_name:
            filter_dict = {"house_name": {"$eq": house_name}}

        print(f"🔍 Querying Pinecone with filter: {filter_dict} and top_k={top_k}")

        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            return results.get("matches", [])
        except Exception as e:
            print(f"❌ Error querying Pinecone: {e}")
            return []

    def extract_text(self, query: str, house_name: str) -> Dict[str, Optional[str]]:
        """Extract relevant text and generate answer using RAG."""
        try:
            # Get query embedding
            query_embedding = self._get_query_embedding(query)
            
            # Query Pinecone
            matches = self._query_pinecone(query_embedding, top_k=20, house_name=house_name)

            if not matches:
                return {'error': 'No relevant matches found'}

            # Combine text from matches
            combined_text = ""
            max_context_length = 4000
            
            for match in matches:
                metadata = match.get('metadata', {})
                text = metadata.get('text', '')[:1500]
                
                if len(combined_text) + len(text) < max_context_length:
                    combined_text += text + "\n\n"
                else:
                    break

            # Clean combined text
            combined_text = re.sub(r'\s+', ' ', combined_text).strip()

            print(f"🔍 Found {len(matches)} matches, combined text length: {len(combined_text)} chars")
            print(f"🧠 Combined Context snippet (first 500 chars):\n{combined_text[:500]}\n--- End snippet ---")
            
            # Debug match information
            print("🔍 Match metadata for debug:")
            for i, match in enumerate(matches[:5]):  # Show first 5 matches
                score = match.get('score', 0)
                metadata = match.get('metadata', {})
                chunk_idx = metadata.get('chunk_index', '?')
                print(f"- Match {i+1}: Chunk {chunk_idx}, Score: {score:.4f}, Metadata keys: {list(metadata.keys())}")

            # Generate response using OpenAI
            prompt = f"""You are a helpful assistant answering questions about housing information from the provided context.

            Use the following extracted document content to answer the user's question accurately.

            Context:
            {combined_text}

            User Question:
            {query}

            Instructions:
            - Provide clear and specific answers based on the context.
            - If the answer is not found in the context, reply: "The document does not specify this information."
            - Be concise but comprehensive in your response.
            """

            response = openai.ChatCompletion.create(
                model=self.chat_model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a professional assistant for answering housing-related queries using provided context. Extract and present information clearly and accurately."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1200,
                top_p=0.9,
            )

            answer = response['choices'][0]['message']['content'].strip()
            
            return {
                "answer": answer,
                "matches_count": len(matches),
                "context_preview": (combined_text[:200] + "...") if len(combined_text) > 200 else combined_text,
                "top_match_score": matches[0].get('score', 0) if matches else 0
            }

        except Exception as e:
            print(f"❌ Error during query: {e}")
            return {'error': str(e)}

    def get_index_stats(self) -> Dict:
        """Get statistics about the Pinecone index."""
        try:
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            print(f"❌ Error getting index stats: {e}")
            return {'error': str(e)}


# Main runner for testing
def main():
    # Load environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = "brindy-house-test-knowladgebase"

    # Validate environment variables
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not found in environment variables")
        return
    
    if not PINECONE_API_KEY:
        print("❌ PINECONE_API_KEY not found in environment variables")
        return

    # Test parameters
    house_name = "Navajo Flats"
    query = "what is the pet policy for navajo flats?"

    try:
        # Initialize RAG query system
        rag_query = PineconeRAGQuery(OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME)
        
        # Get index statistics
        print("📊 Index Statistics:")
        stats = rag_query.get_index_stats()
        if 'error' not in stats:
            print(f"   Total vectors: {stats.get('total_vector_count', 'Unknown')}")
            print(f"   Namespaces: {list(stats.get('namespaces', {}).keys())}")
        
        # Perform query
        print(f"\n🔍 Searching for: '{query}'")
        print(f"🏠 House filter: '{house_name}'")
        
        result = rag_query.extract_text(query, house_name)

        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"\n✅ Answer: {result['answer']}")
            print(f"📊 Matches found: {result['matches_count']}")
            print(f"🎯 Top match score: {result.get('top_match_score', 'N/A')}")
            print(f"📝 Context preview: {result['context_preview']}")

        return result

    except Exception as e:
        print(f"❌ Main execution error: {e}")
        return {'error': str(e)}


if __name__ == "__main__":
    main()