# import os
# import re
# from typing import List, Dict
# import pytesseract
# from PIL import Image
# import io
# from docx import Document
# from pinecone import Pinecone, ServerlessSpec
# import google.generativeai as genai
# import docx2txt
# import fitz  # PyMuPDF
# from dotenv import load_dotenv

# class WordVectorizerGeminiPinecone:
#     def __init__(
#         self,
#         file_path: str,
#         pinecone_index_name: str,
#         house_name: str,
#         PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY'),
#         GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY'),
#     ):
#         """
#         Initialize the Word document vectorizer with Google Gemini embeddings and Pinecone storage.
#         If delete_existing_index=True, deletes the current index before embedding new data.
#         """
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"File not found at: {file_path}")
#         self.file_path = file_path
#         self.index_name = pinecone_index_name
#         self.house_name = house_name  # Manually set house name

#         # Google Gemini setup
#         self.embedding_model = "models/text-embedding-004"
#         genai.configure(api_key=GOOGLE_API_KEY)

#         # Pinecone setup
#         self.pc = Pinecone(api_key=PINECONE_API_KEY)


#         if pinecone_index_name not in [i.name for i in self.pc.list_indexes()]:
#             print(f"Creating new index: {pinecone_index_name}")
#             self.pc.create_index(
#                 name=pinecone_index_name,
#                 dimension=768,
#                 metric="cosine",
#                 spec=ServerlessSpec(cloud="aws", region="us-east-1")
#             )

#         self.index = self.pc.Index(pinecone_index_name)

#     def extract_text_from_document(self, doc_path: str) -> str:
#         """Extract all text from a document (PDF or DOCX), including OCR from images."""
#         try:
#             if doc_path.lower().endswith('.pdf'):
#                 # Handle PDF files
#                 doc = fitz.open(doc_path)
#                 text_content = []
#                 for page in doc:
#                     text_content.append(page.get_text())
#                 doc_text = "\n\n".join(text_content)
                
#                 # Extract images and perform OCR
#                 ocr_texts = self.extract_images_and_ocr(doc_path)
                
#                 # Combine text and OCR results
#                 full_text = doc_text + "\n\n" + "\n\n".join(ocr_texts)
#                 return full_text.strip()
                
#             elif doc_path.lower().endswith('.docx'):
#                 # Handle DOCX files (existing functionality)
#                 doc_text = docx2txt.process(doc_path)
#                 ocr_texts = self.extract_images_and_ocr(doc_path)
#                 full_text = doc_text + "\n\n" + "\n\n".join(ocr_texts)
#                 return full_text.strip()
#             else:
#                 raise ValueError(f"Unsupported file type: {doc_path}")
                
#         except Exception as e:
#             print(f"Error extracting text from {doc_path}: {e}")
#             return ""

#     def extract_images_and_ocr(self, doc_path: str) -> List[str]:
#         """Extract images from document and perform OCR."""
#         ocr_texts = []
#         try:
#             doc = fitz.open(doc_path)
#             for page_num in range(len(doc)):
#                 page = doc.load_page(page_num)
#                 image_list = page.get_images(full=True)
                
#                 # Also check for images in PDF format
#                 if doc_path.lower().endswith('.pdf'):
#                     pix = page.get_pixmap()
#                     img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#                     try:
#                         text = pytesseract.image_to_string(img)
#                         if text.strip():
#                             ocr_texts.append(f"[Image OCR Text Page {page_num+1}]: {text.strip()}")
#                     except Exception as e:
#                         print(f"OCR error on page {page_num}: {e}")
                
#                 # Process embedded images
#                 for img_idx, img_info in enumerate(image_list):
#                     xref = img_info[0]
#                     base_image = doc.extract_image(xref)
#                     image_bytes = base_image["image"]
#                     image = Image.open(io.BytesIO(image_bytes))
#                     try:
#                         text = pytesseract.image_to_string(image)
#                         if text.strip():
#                             ocr_texts.append(f"[Image OCR Text Page {page_num+1}]: {text.strip()}")
#                     except Exception as e:
#                         print(f"OCR error on image {img_idx} of page {page_num}: {e}")
                        
#         except Exception as e:
#             print(f"Error extracting images from {doc_path}: {e}")
            
#         return ocr_texts

#     def create_embedding(self, text: str) -> List[float]:
#         truncated = text[:8000]
#         result = genai.embed_content(
#             model=self.embedding_model,
#             content=truncated,
#             task_type="retrieval_document"
#         )
#         return result["embedding"]

#     def embed_and_store(self, doc_path: str):
#         """
#         Process a single document and create a single embedding vector for the entire document.
#         """
#         file_name = os.path.basename(doc_path)
        
#         # Extract all text from document including OCR of images
#         full_text = self.extract_text_from_document(doc_path)
        
#         if not full_text:
#             print(f"Warning: No text extracted from {file_name}")
#             return
            
#         try:
#             # Create embedding for the entire document
#             embedding = self.create_embedding(full_text)
#             vector_id = self.house_name  # Use the provided house name for vector ID
            
#             metadata = {
#                 "file_name": file_name,
#                 "house_name": self.house_name,  # Use the provided house name
#                 "text": full_text[:8000],  # Truncate for metadata
#                 "char_count": len(full_text)
#             }

#             # Check if the house name already exists in the index
#             existing_query = self.index.query(
#                 vector=self.create_embedding(self.house_name),
#                 top_k=1,
#                 include_metadata=True
#             )
#             if existing_query["matches"]:
#                 print(f"Replacing existing file for house: {self.house_name}")
#                 self.index.delete(ids=[vector_id])  # Delete the existing vector
            
#             # Upload the new vector for the house
#             self.index.upsert(vectors=[{
#                 "id": vector_id,
#                 "values": embedding,
#                 "metadata": metadata
#             }])
            
#             print(f"Uploaded vector for house: {vector_id}")
            
#         except Exception as e:
#             print(f"Error embedding document {file_name}: {e}")

#     # def process(self):
#     #     print(f"Processing {self.file_path}")
#     #     self.embed_and_store(self.file_path)

#     def query_similar(self, query_text: str, top_k: int = 5):
#         query_embedding = self.create_embedding(query_text)
#         results = self.index.query(
#             vector=query_embedding,
#             top_k=top_k,
#             include_metadata=True
#         )
#         return results


# # Example usage
# if __name__ == "__main__":
#     # FILE_PATH = "D:\\brindyjean\\House Notes (Internal)\\Req (1).pdf"
#     # PINECONE_INDEX_NAME = "house-information-embeddings"
#     # HOUSE_NAME = "Test House"  # Example house name
    
#     vectorizer = WordVectorizerGeminiPinecone(
#         file_path="D:\\brindyjean\\House Notes (Internal)\\HN_ 81st Way Desert Rose.docx",
#         pinecone_index_name="house-information-embeddings",
#         house_name="81st Way Desert Rose",
#     )
    
#     # Process the given document
#     print(f"Processing {vectorizer.file_path}")
#     vectorizer.embed_and_store(vectorizer.file_path)






import os
import re
from typing import List, Dict, Tuple
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
        chunk_size: int = 6000,  # Leave some buffer for embedding model
        chunk_overlap: int = 5000,  # Overlap between chunks for context continuity
        PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY'),
        GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY'),
    ):
        """
        Initialize the Word document vectorizer with Google Gemini embeddings and Pinecone storage.
        Now supports chunking for large documents.
        
        Args:
            chunk_size: Maximum characters per chunk (default 6000 to leave buffer)
            chunk_overlap: Characters to overlap between chunks for context continuity
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")
        self.file_path = file_path
        self.index_name = pinecone_index_name
        self.house_name = house_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Google Gemini setup
        self.embedding_model = "models/text-embedding-004"
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
        """Create embedding without truncation - assumes text is already properly sized"""
        result = genai.embed_content(
            model=self.embedding_model,
            content=text,
            task_type="retrieval_document"
        )
        return result["embedding"]

    def chunk_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into overlapping chunks.
        
        Returns:
            List of tuples: (chunk_text, start_position, end_position)
        """
        if len(text) <= self.chunk_size:
            return [(text, 0, len(text))]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If this isn't the last chunk, try to break at a sentence or paragraph
            if end < len(text):
                # Look for sentence endings within the last 200 characters
                search_start = max(end - 200, start + self.chunk_size // 2)
                sentence_break = max(
                    text.rfind('.', search_start, end),
                    text.rfind('!', search_start, end),
                    text.rfind('?', search_start, end),
                    text.rfind('\n\n', search_start, end)
                )
                
                if sentence_break > search_start:
                    end = sentence_break + 1
            
            chunk_text = text[start:end].strip()
            if chunk_text:  # Only add non-empty chunks
                chunks.append((chunk_text, start, end))
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start >= end:
                start = end
                
        return chunks

    def delete_existing_house_vectors(self):
        """Delete all existing vectors for this house name"""
        try:
            # Query for all vectors with this house name
            results = self.index.query(
                vector=[0.0] * 768,  # Dummy vector
                top_k=10000,  # Large number to get all
                include_metadata=True,
                filter={"house_name": self.house_name}
            )
            
            if results["matches"]:
                vector_ids = [match["id"] for match in results["matches"]]
                self.index.delete(ids=vector_ids)
                print(f"Deleted {len(vector_ids)} existing vectors for house: {self.house_name}")
            else:
                print(f"No existing vectors found for house: {self.house_name}")
                
        except Exception as e:
            print(f"Error deleting existing vectors: {e}")

    def embed_and_store(self, doc_path: str):
        """
        Process a document and create multiple embedding vectors if needed.
        Each chunk gets stored as a separate vector with the same house name.
        """
        file_name = os.path.basename(doc_path)
        
        # Extract all text from document including OCR of images
        full_text = self.extract_text_from_document(doc_path)
        
        if not full_text:
            print(f"Warning: No text extracted from {file_name}")
            return
            
        print(f"Document length: {len(full_text)} characters")
        
        # Delete existing vectors for this house
        self.delete_existing_house_vectors()
        
        # Chunk the text
        chunks = self.chunk_text(full_text)
        print(f"Created {len(chunks)} chunks for processing")
        
        vectors_to_upsert = []
        
        for i, (chunk_text, start_pos, end_pos) in enumerate(chunks):
            try:
                # Create embedding for this chunk
                embedding = self.create_embedding(chunk_text)
                
                # Create unique vector ID for this chunk
                vector_id = f"{self.house_name}_chunk_{i+1}"
                
                metadata = {
                    "file_name": file_name,
                    "house_name": self.house_name,
                    "chunk_index": i + 1,
                    "total_chunks": len(chunks),
                    "text": chunk_text[:1000],  # Store first 1000 chars for preview
                    "full_text_start": start_pos,
                    "full_text_end": end_pos,
                    "chunk_char_count": len(chunk_text),
                    "total_document_chars": len(full_text)
                }

                vectors_to_upsert.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })
                
                print(f"Prepared chunk {i+1}/{len(chunks)} ({len(chunk_text)} chars)")
                
            except Exception as e:
                print(f"Error embedding chunk {i+1}: {e}")
                continue
        
        # Batch upsert all vectors
        if vectors_to_upsert:
            try:
                self.index.upsert(vectors=vectors_to_upsert)
                print(f"Successfully uploaded {len(vectors_to_upsert)} vectors for house: {self.house_name}")
            except Exception as e:
                print(f"Error uploading vectors: {e}")

    def query_similar(self, query_text: str, top_k: int = 5, house_filter: str = None):
        """
        Query for similar content. Can optionally filter by house name.
        
        Args:
            query_text: Text to search for
            top_k: Number of results to return
            house_filter: Optional house name to filter results
        """
        query_embedding = self.create_embedding(query_text)
        
        query_params = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True
        }
        
        # Add house filter if specified
        if house_filter:
            query_params["filter"] = {"house_name": house_filter}
        
        results = self.index.query(**query_params)
        return results

    def get_all_chunks_for_house(self, house_name: str = None) -> Dict:
        """
        Retrieve all chunks for a specific house (or current house if not specified).
        """
        target_house = house_name or self.house_name
        
        try:
            results = self.index.query(
                vector=[0.0] * 768,  # Dummy vector
                top_k=10000,  # Large number to get all chunks
                include_metadata=True,
                filter={"house_name": target_house}
            )
            
            if results["matches"]:
                # Sort chunks by chunk_index
                sorted_chunks = sorted(results["matches"], 
                                     key=lambda x: x["metadata"].get("chunk_index", 0))
                return {
                    "house_name": target_house,
                    "total_chunks": len(sorted_chunks),
                    "chunks": sorted_chunks
                }
            else:
                return {"house_name": target_house, "total_chunks": 0, "chunks": []}
                
        except Exception as e:
            print(f"Error retrieving chunks for house {target_house}: {e}")
            return {"house_name": target_house, "total_chunks": 0, "chunks": []}

    def get_document_stats(self) -> Dict:
        """Get statistics about the processed document"""
        try:
            chunks_info = self.get_all_chunks_for_house()
            if chunks_info["chunks"]:
                first_chunk = chunks_info["chunks"][0]["metadata"]
                return {
                    "house_name": self.house_name,
                    "file_name": first_chunk.get("file_name", "Unknown"),
                    "total_chunks": chunks_info["total_chunks"],
                    "total_document_chars": first_chunk.get("total_document_chars", 0),
                    "chunk_size_used": self.chunk_size,
                    "chunk_overlap_used": self.chunk_overlap
                }     
            else:
                return {"error": "No chunks found for this house"}
        except Exception as e:
            return {"error": f"Error getting document stats: {e}"}


# Example usage
if __name__ == "__main__":
    # Example with chunking
    vectorizer = WordVectorizerGeminiPinecone(
        file_path="D:\\brindyjean\\House Notes PDF\\HN_ Siesta Pacifica.pdf",
        pinecone_index_name="house-information-embeddings",
        house_name="Siesta Pacifica",
        chunk_size=6000,  # 6000 characters per chunk
        chunk_overlap=500   # 200 character overlap
    )
    
    # Process the document with chunking
    print(f"Processing {vectorizer.file_path}")
    vectorizer.embed_and_store(vectorizer.file_path)
    
    # Get statistics about the processing
    stats = vectorizer.get_document_stats()
    print(f"Document Stats: {stats}")
    
