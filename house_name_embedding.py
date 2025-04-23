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
from dotenv import load_dotenv

class WordVectorizerGeminiPinecone:
    def __init__(
        self,
        file_path: str,
        pinecone_index_name: str,
        house_name: str,
        PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY'),
        GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY'),
    ):
        """
        Initialize the Word document vectorizer with Google Gemini embeddings and Pinecone storage.
        If delete_existing_index=True, deletes the current index before embedding new data.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")
        self.file_path = file_path
        self.index_name = pinecone_index_name
        self.house_name = house_name  # Manually set house name

        # Google Gemini setup
        self.embedding_model = "models/embedding-001"
        genai.configure(api_key=GOOGLE_API_KEY)

        # Pinecone setup
        self.pc = Pinecone(api_key=PINECONE_API_KEY)


        if pinecone_index_name not in [i.name for i in self.pc.list_indexes()]:
            print(f"Creating new index: {pinecone_index_name}")
            self.pc.create_index(
                name=pinecone_index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        self.index = self.pc.Index(pinecone_index_name)

    def extract_text_from_document(self, doc_path: str) -> str:
        """Extract all text from a document (PDF or DOCX), including OCR from images."""
        try:
            if doc_path.lower().endswith('.pdf'):
                # Handle PDF files
                doc = fitz.open(doc_path)
                text_content = []
                for page in doc:
                    text_content.append(page.get_text())
                doc_text = "\n\n".join(text_content)
                
                # Extract images and perform OCR
                ocr_texts = self.extract_images_and_ocr(doc_path)
                
                # Combine text and OCR results
                full_text = doc_text + "\n\n" + "\n\n".join(ocr_texts)
                return full_text.strip()
                
            elif doc_path.lower().endswith('.docx'):
                # Handle DOCX files (existing functionality)
                doc_text = docx2txt.process(doc_path)
                ocr_texts = self.extract_images_and_ocr(doc_path)
                full_text = doc_text + "\n\n" + "\n\n".join(ocr_texts)
                return full_text.strip()
            else:
                raise ValueError(f"Unsupported file type: {doc_path}")
                
        except Exception as e:
            print(f"Error extracting text from {doc_path}: {e}")
            return ""

    def extract_images_and_ocr(self, doc_path: str) -> List[str]:
        """Extract images from document and perform OCR."""
        ocr_texts = []
        try:
            doc = fitz.open(doc_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images(full=True)
                
                # Also check for images in PDF format
                if doc_path.lower().endswith('.pdf'):
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    try:
                        text = pytesseract.image_to_string(img)
                        if text.strip():
                            ocr_texts.append(f"[Image OCR Text Page {page_num+1}]: {text.strip()}")
                    except Exception as e:
                        print(f"OCR error on page {page_num}: {e}")
                
                # Process embedded images
                for img_idx, img_info in enumerate(image_list):
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
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
        
        # Extract all text from document including OCR of images
        full_text = self.extract_text_from_document(doc_path)
        
        if not full_text:
            print(f"Warning: No text extracted from {file_name}")
            return
            
        try:
            # Create embedding for the entire document
            embedding = self.create_embedding(full_text)
            vector_id = self.house_name  # Use the provided house name for vector ID
            
            metadata = {
                "file_name": file_name,
                "house_name": self.house_name,  # Use the provided house name
                "text": full_text[:8000],  # Truncate for metadata
                "char_count": len(full_text)
            }

            # Check if the house name already exists in the index
            existing_query = self.index.query(
                vector=self.create_embedding(self.house_name),
                top_k=1,
                include_metadata=True
            )
            if existing_query["matches"]:
                print(f"Replacing existing file for house: {self.house_name}")
                self.index.delete(ids=[vector_id])  # Delete the existing vector
            
            # Upload the new vector for the house
            self.index.upsert(vectors=[{
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            }])
            
            print(f"Uploaded vector for house: {vector_id}")
            
        except Exception as e:
            print(f"Error embedding document {file_name}: {e}")

    # def process(self):
    #     print(f"Processing {self.file_path}")
    #     self.embed_and_store(self.file_path)

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
    # FILE_PATH = "D:\\brindyjean\\House Notes (Internal)\\Req (1).pdf"
    # PINECONE_INDEX_NAME = "house-information-embeddings"
    # HOUSE_NAME = "Test House"  # Example house name
    
    vectorizer = WordVectorizerGeminiPinecone(
        file_path="D:\\brindyjean\\House Notes (Internal)\\HN_ Elmerville Hummingbird Crossing.docx",
        pinecone_index_name="house-information-embeddings",
        house_name="Elmerville Hummingbird Crossing",
    )
    
    # Process the given document
    print(f"Processing {vectorizer.file_path}")
    vectorizer.embed_and_store(vectorizer.file_path)
