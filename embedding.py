import os
import json
import numpy as np
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from typing import List, Dict
from docx import Document

class WordVectorizerGeminiPinecone:
    def __init__(self, folder_path: str, google_api_key: str, pinecone_api_key: str, pinecone_index_name: str):
        """
        Initialize the Word document vectorizer with Google Gemini embeddings and Pinecone storage.
        
        Args:
            folder_path: Path to folder containing Word documents
            google_api_key: Google API key for Gemini
            pinecone_api_key: Pinecone API key
            pinecone_index_name: Name of the Pinecone index to use
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found at: {folder_path}")
        self.folder_path = folder_path
        
        # Initialize Google Gemini
        self.embedding_model = "models/embedding-001"
        genai.configure(api_key=google_api_key)
        
        # Initialize Pinecone with updated API
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        # Create index if it doesn't exist (dimension 768 for Gemini embedding-001)
        if pinecone_index_name not in [index.name for index in self.pc.list_indexes()]:
            self.pc.create_index(
                name=pinecone_index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )  # Adjust cloud/region as needed
            )
            
        self.index = self.pc.Index(pinecone_index_name)
    
    def extract_text_by_section(self, doc_path: str) -> Dict[int, str]:
        """
        Approximate 'page-by-page' sectioning by splitting on empty paragraphs.
        
        Args:
            doc_path: Path to the Word document
            
        Returns:
            Dictionary mapping section numbers to section text
        """
        doc = Document(doc_path)
        section_texts = {}
        current_text = ""
        section_index = 1
        
        for para in doc.paragraphs:
            if para.text.strip():
                current_text += para.text + "\n"
            else:
                if current_text.strip():
                    section_texts[section_index] = current_text.strip()
                    current_text = ""
                    section_index += 1
                    
        # Don't forget the last section if it exists
        if current_text.strip():
            section_texts[section_index] = current_text.strip()
            
        print(f"Extracted {len(section_texts)} sections from {doc_path}")
        return section_texts
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Create embedding for text using Google Gemini embedding model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Truncate text to Gemini's limit
        truncated = text[:8000]
        
        result = genai.embed_content(
            model=self.embedding_model,
            content=truncated,
            task_type="retrieval_document"
        )
        
        return result["embedding"]
    
    def embed_and_store(self, doc_path: str):
        """
        Extract text sections from document, create embeddings, and store in Pinecone.
        
        Args:
            doc_path: Path to the Word document
        """
        sections = self.extract_text_by_section(doc_path)
        file_name = os.path.basename(doc_path)
        
        for section_num, content in sections.items():
            try:
                # Create embedding
                embedding = self.create_embedding(content)
                
                # Create unique ID and metadata
                vector_id = f"{file_name}_{section_num}"
                
                # Truncate for metadata storage (if needed)
                truncated = content[:8000]  # Gemini limit
                
                metadata = {
                    "file_name": file_name,
                    "section_number": section_num,
                    "text": truncated,
                    "char_count": len(truncated)
                }
                
                # Store in Pinecone with updated API
                self.index.upsert(
                    vectors=[
                        {
                            "id": vector_id,
                            "values": embedding,
                            "metadata": metadata
                        }
                    ]
                )
                print(f"Embedded and uploaded: {vector_id}")
                
            except Exception as e:
                print(f"Error embedding section {section_num} of {file_name}: {str(e)}")
    
    def process_all(self):
        """
        Process all Word documents in the specified folder
        """
        for file_name in os.listdir(self.folder_path):
            if file_name.lower().endswith(".docx"):
                file_path = os.path.join(self.folder_path, file_name)
                print(f"Processing {file_path}")
                self.embed_and_store(file_path)
    
    def query_similar(self, query_text: str, top_k: int = 5):
        """
        Query Pinecone for documents similar to the query text
        
        Args:
            query_text: Text to find similar documents for
            top_k: Number of results to return
            
        Returns:
            Dictionary of similar documents with metadata
        """
        query_embedding = self.create_embedding(query_text)
        
        # Updated query syntax for new Pinecone API
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return results

# Example usage
if __name__ == "__main__":
    FOLDER_PATH = "D:\\brindyjean\\Guest Messaging Guide"
    GOOGLE_API_KEY = "REMOVED"
    PINECONE_API_KEY = "pcsk_53KbzW_9Qc9oTo42j16gSVRoghPig376R4oKMnkafv8MatjWeCCZhouKEioasKejD1zv5b"
    PINECONE_INDEX_NAME = "guest-messaging-guide-doc-embeddings"
    
    vectorizer = WordVectorizerGeminiPinecone(FOLDER_PATH, GOOGLE_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME)
    
    # Process all documents
    vectorizer.process_all()
    
    # Example query
    # similar_docs = vectorizer.query_similar("What is the contract termination clause?")
    # for match in similar_docs["matches"]:
    #     print(f"Score: {match['score']}")
    #     print(f"Document: {match['metadata']['file_name']}, Section: {match['metadata']['section_number']}")
    #     print(f"Text: {match['metadata']['text'][:200]}...")
    #     print("-" * 50)

