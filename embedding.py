
import os
import re
from typing import List, Dict
import pytesseract
from PIL import Image
import io
from docx import Document
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
import docx2txt
import fitz  # PyMuPDF


class WordVectorizerGeminiPinecone:
    def __init__(
        self,
        folder_path: str,
        google_api_key: str,
        pinecone_api_key: str,
        pinecone_index_name: str,
        delete_existing_index: bool = False
    ):
        """
        Initialize the Word document vectorizer with Google Gemini embeddings and Pinecone storage.
        If delete_existing_index=True, deletes the current index before embedding new data.
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found at: {folder_path}")
        self.folder_path = folder_path
        self.index_name = pinecone_index_name

        # Google Gemini setup
        self.embedding_model = "models/embedding-001"
        genai.configure(api_key=google_api_key)

        # Pinecone setup
        self.pc = Pinecone(api_key=pinecone_api_key)

        # Delete and recreate index if flag is set
        if delete_existing_index and pinecone_index_name in [i.name for i in self.pc.list_indexes()]:
            print(f"Deleting existing index: {pinecone_index_name}")
            self.pc.delete_index(pinecone_index_name)

        if pinecone_index_name not in [i.name for i in self.pc.list_indexes()]:
            print(f"Creating new index: {pinecone_index_name}")
            self.pc.create_index(
                name=pinecone_index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        self.index = self.pc.Index(pinecone_index_name)

    def extract_house_name(self, file_name: str) -> str:
        """
        Extract house name from filename using improved pattern matching.
        Handles various formats like "HN_Navajo Flats.docx", "BRINDY NOTES - Navajo Flats.docx", etc.
        """
        # First try to match the HN pattern
        hn_match = re.search(r"HN[_\- ]*([\w\s]+?)(?:\.docx)?$", file_name, re.IGNORECASE)
        if hn_match:
            return hn_match.group(1).strip()
        
        # Then try to match the pattern with "NOTES - [House Name]"
        notes_match = re.search(r"NOTES\s*-\s*([\w\s]+?)(?:\.docx)?$", file_name, re.IGNORECASE)
        if notes_match:
            return notes_match.group(1).strip()
        
        # If no clear house name, return the filename without extension
        base_name = os.path.splitext(file_name)[0]
        return base_name.strip()

    def extract_text_from_document(self, doc_path: str) -> str:
        """
        Extract all text from a Word document, including processing embedded images with OCR.
        Returns a single string with all document content.
        """
        # Extract text content from document
        try:
            doc_text = docx2txt.process(doc_path)
            
            # Process images with OCR
            ocr_texts = self.extract_images_and_ocr(doc_path)
            
            # Combine text and OCR results
            full_text = doc_text + "\n\n" + "\n\n".join(ocr_texts)
            return full_text.strip()
            
        except Exception as e:
            print(f"Error extracting text from {doc_path}: {e}")
            return ""

    def extract_images_and_ocr(self, doc_path: str) -> List[str]:
        """
        Extract images from Word document and perform OCR on them.
        Returns a list of text extracted from images.
        """
        ocr_texts = []
        
        try:
            # Extract images using PyMuPDF
            doc = fitz.open(doc_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images(full=True)
                
                for img_idx, img_info in enumerate(image_list):
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Use PIL and pytesseract for OCR
                    image = Image.open(io.BytesIO(image_bytes))
                    try:
                        text = pytesseract.image_to_string(image)
                        if text.strip():
                            ocr_texts.append(f"[Image OCR Text Page {page_num+1}]: {text.strip()}")
                    except Exception as e:
                        print(f"OCR error on image {img_idx} of page {page_num}: {e}")
                        
        except Exception as e:
            print(f"Error extracting images from {doc_path}: {e}")
            
        return ocr_texts

    def create_embedding(self, text: str) -> List[float]:
        truncated = text[:8000]
        result = genai.embed_content(
            model=self.embedding_model,
            content=truncated,
            task_type="retrieval_document"
        )
        return result["embedding"]

    def embed_and_store(self, doc_path: str):
        """
        Process a single document and create a single embedding vector for the entire document.
        """
        file_name = os.path.basename(doc_path)
        house_name = self.extract_house_name(file_name)
        
        # Extract all text from document including OCR of images
        full_text = self.extract_text_from_document(doc_path)
        
        if not full_text:
            print(f"Warning: No text extracted from {file_name}")
            return
            
        try:
            # Create embedding for the entire document
            embedding = self.create_embedding(full_text)
            vector_id = file_name
            
            metadata = {
                "file_name": file_name,
                "house_name": house_name,
                "text": full_text[:8000],  # Truncate for metadata
                "char_count": len(full_text)
            }
            
            self.index.upsert(vectors=[{
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            }])
            
            print(f"Uploaded vector for entire document: {vector_id}")
            
        except Exception as e:
            print(f"Error embedding document {file_name}: {e}")

    def process_all(self):
        for file_name in os.listdir(self.folder_path):
            if file_name.lower().endswith(".docx"):
                file_path = os.path.join(self.folder_path, file_name)
                print(f"Processing {file_path}")
                self.embed_and_store(file_path)

    def query_similar(self, query_text: str, top_k: int = 5):
        query_embedding = self.create_embedding(query_text)
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
    
    vectorizer = WordVectorizerGeminiPinecone(
    folder_path=FOLDER_PATH,
    google_api_key=GOOGLE_API_KEY,
    pinecone_api_key=PINECONE_API_KEY,
    pinecone_index_name=PINECONE_INDEX_NAME,
    delete_existing_index=True  # 🔥 Deletes the existing index before proceeding
    )
    
    # Process all documents
    vectorizer.process_all()