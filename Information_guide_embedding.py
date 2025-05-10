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
        file_path: str,
        pinecone_index_name: str,
        chunk_size: int = 8000,  # Max characters per chunk
        delete_existing_index: bool = False,
        PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY'),
        GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY'),
    ):
        """
        Initialize the Word document vectorizer with Google Gemini embeddings and Pinecone storage.
        Supports chunking of document text into multiple embeddings.
        """
        if not os.path.isfile(file_path):  # Changed from exists to isfile
            raise FileNotFoundError(f"File not found at: {file_path}")
        self.file_path = file_path
        self.index_name = pinecone_index_name
        self.chunk_size = chunk_size

        # Google Gemini setup
        self.embedding_model = "models/text-embedding-004"
        genai.configure(api_key=GOOGLE_API_KEY)

        # Pinecone setup
        self.pc = Pinecone(api_key=PINECONE_API_KEY)

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

    def split_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks of specified size."""
        chunks = []
        current_chunk = ""
        
        for line in text.split("\n"):
            if len(current_chunk) + len(line) + 1 <= chunk_size:
                current_chunk += line + "\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = line + "\n"
                
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        # If any chunk exceeds chunk_size due to long lines, split further
        final_chunks = []
        for chunk in chunks:
            while len(chunk) > chunk_size:
                final_chunks.append(chunk[:chunk_size].strip())
                chunk = chunk[chunk_size:]
            if chunk.strip():
                final_chunks.append(chunk.strip())
                
        return final_chunks

    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text using Google Gemini."""
        truncated = text[:8000]  # Ensure no chunk exceeds Gemini limit
        result = genai.embed_content(
            model=self.embedding_model,
            content=truncated,
            task_type="retrieval_document"
        )
        return result["embedding"]

    def embed_and_store(self, doc_path: str):
        """Process a document, split into chunks, and embed each chunk."""
        file_name = os.path.basename(doc_path)
        
        # Extract all text including OCR
        full_text = self.extract_text_from_document(doc_path)
        
        if not full_text:
            print(f"Warning: No text extracted from {file_name}")
            return
            
        # Split into chunks
        chunks = self.split_into_chunks(full_text, self.chunk_size)
        
        for chunk_num, chunk_text in enumerate(chunks, 1):
            try:
                # Create embedding for this chunk
                embedding = self.create_embedding(chunk_text)
                vector_id = f"{file_name}_chunk_{chunk_num}"
                
                metadata = {
                    "file_name": file_name,
                    "title": "Information & Massageing Guide",
                    "chunk_number": chunk_num,
                    "text": chunk_text[:8000],  # Truncate for metadata
                    "char_count": len(chunk_text),
                    "total_chunks": len(chunks)
                }
                
                self.index.upsert(vectors=[{
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                }])
                
                print(f"Uploaded vector for {vector_id} ({len(chunk_text)} chars)")
                
            except Exception as e:
                print(f"Error embedding chunk {chunk_num} of {file_name}: {e}")

    # def process_all(self):
    #     """Process all Word documents in the folder."""
    #     for file_name in os.listdir(self.file_path):
    #         if file_name.lower().endswith(".docx"):
    #             file_path = os.path.join(self.file_path, file_name)
    #             print(f"Processing {file_path}")
    #             self.embed_and_store(file_path)

    def query_similar(self, query_text: str, top_k: int = 5):
        """Query Pinecone for similar chunks."""
        query_embedding = self.create_embedding(query_text)
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results


# Example usage
if __name__ == "__main__":
    # FILE_PATH = "D:\\brindyjean\\Guest Messaging Guide\\Guest Messaging Guide Organized.pdf"
    # PINECONE_INDEX_NAME = "information-massageing-guide-embeddings"
    
    vectorizer = WordVectorizerGeminiPinecone(
        file_path="D:\\brindyjean\\Guest Messaging Guide\\Guest Messaging Guide (Brandyjeans).docx",
        pinecone_index_name="information-massageing-guide-embeddings",
        delete_existing_index=True,
        chunk_size=2000  # Adjust chunk size as needed
    )
    
    # Process all documents
    print(f"Processing {vectorizer.file_path}")
    vectorizer.embed_and_store(vectorizer.file_path)