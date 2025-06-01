
import random
import string
import os
import fitz  # PyMuPDF
from typing import List, Dict
import openai
import pytesseract
from PIL import Image
import io
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import re

load_dotenv()

class WordVectorizerOpenAIPinecone:
    def __init__(
        self,
        folder_path: str,
        pinecone_index_name: str,
    ):
        """
        Initialize the PDF vectorizer with OpenAI embeddings and Pinecone storage.
        Processes PDF files from the specified folder path, using the PDF name as part of the vector ID.
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found at: {folder_path}")
        self.folder_path = folder_path
        self.index_name = pinecone_index_name

        # OpenAI setup
        self.embedding_model = "text-embedding-ada-002"
        openai.api_key = os.getenv("OPENAI_API_KEY")

        # Pinecone setup
        self.pc = Pinecone(os.environ.get('PINECONE_API_KEY'))

        if pinecone_index_name not in [i.name for i in self.pc.list_indexes()]:
            print(f"Creating new index: {pinecone_index_name}")
            self.pc.create_index(
                name=pinecone_index_name,
                dimension=1536,  # Dimension for text-embedding-3-small
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        self.index = self.pc.Index(pinecone_index_name)

    def generate_unique_id(self) -> str:
        """Generate a 5-character unique alphanumeric ID."""
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))

    def sanitize_vector_id(self, vector_id: str) -> str:
        """Sanitize the vector ID to ensure it contains only ASCII characters and is valid for Pinecone.
        Replaces spaces, parentheses, and other non-alphanumeric characters with underscores."""
        # Replace spaces, parentheses, and other special characters with underscores
        sanitized = re.sub(r'[^\w\d]', '_', vector_id)
        # Ensure it's ASCII only by replacing any remaining non-ASCII characters
        sanitized = re.sub(r'[^\x00-\x7F]+', '_', sanitized)
        # Truncate if too long (Pinecone has limits on ID length)
        if len(sanitized) > 64:
            # Keep a prefix and add a hash-like suffix for uniqueness
            prefix = sanitized[:54]  # Leave room for unique suffix
            suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
            sanitized = f"{prefix}_{suffix}"
        return sanitized

    def extract_text_from_pdf_page(self, doc: fitz.Document, page_num: int) -> str:
        """Extract text from a single PDF page, including OCR from images."""
        try:
            page = doc.load_page(page_num)
            text_content = page.get_text().strip()

            # Perform OCR on the page as an image
            ocr_texts = []
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            try:
                ocr_text = pytesseract.image_to_string(img).strip()
                if ocr_text:
                    ocr_texts.append(ocr_text)
            except Exception as e:
                print(f"OCR error on page {page_num + 1}: {e}")

            # Process embedded images
            image_list = page.get_images(full=True)
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                try:
                    ocr_text = pytesseract.image_to_string(image).strip()
                    if ocr_text:
                        ocr_texts.append(ocr_text)
                except Exception as e:
                    print(f"OCR error on image {img_idx} of page {page_num + 1}: {e}")

            # Combine page text and OCR results
            full_text = text_content + "\n\n" + "\n\n".join(ocr_texts)
            return full_text.strip()

        except Exception as e:
            print(f"Error extracting text from page {page_num + 1}: {e}")
            return ""

    def create_embedding(self, text: str) -> List[float]:
        """Create an embedding for the given text using OpenAI's embedding model."""
        try:
            # Truncate text to avoid exceeding token limits (approx. 8192 tokens for text-embedding-3-small)
            truncated = text[:8000]
            response = openai.Embedding.create(
                model=self.embedding_model,
                input=truncated
            )
            return response['data'][0]['embedding']
        except Exception as e:
            print(f"Error creating embedding: {e}")
            return []

    def delete_vectors_by_pdf_name(self, pdf_name: str):
        """Delete all existing vectors in the index related to a specific PDF name."""
        try:
            # First, query to get the IDs of vectors associated with this PDF name
            results = self.index.query(
                vector=[0.0] * 1536,  # Dummy vector for metadata-only search
                top_k=10000,  # Set to a high value to get all matches
                include_metadata=True,
                filter={"pdf_name": {"$eq": pdf_name}}
            )
            
            # Extract vector IDs
            if results and "matches" in results:
                vector_ids = [match["id"] for match in results["matches"] if "id" in match]
                
                if vector_ids:
                    # Delete the vectors in batches to avoid potential limitations
                    batch_size = 100
                    for i in range(0, len(vector_ids), batch_size):
                        batch = vector_ids[i:i+batch_size]
                        self.index.delete(ids=batch)
                    
                    print(f"Deleted {len(vector_ids)} vectors for PDF: {pdf_name}")
                else:
                    print(f"No vectors found for PDF: {pdf_name}")
            else:
                print(f"No matches found for PDF: {pdf_name}")
                
        except Exception as e:
            print(f"Error deleting vectors for PDF {pdf_name}: {e}")
            # Continue with processing rather than failing completely

    def embed_and_store_pdf(self, file_path: str, house_name: str, replace_existing: bool = True):
        """
        Process a single PDF document page by page and create embeddings for each page.
        Store embeddings in Pinecone with a unique ID for each page.
        If replace_existing is True, delete existing vectors for the same PDF before adding new ones.
        """
        pdf_name = os.path.splitext(os.path.basename(file_path))[0]  # Extract PDF name without extension
        
        try:
            # If replacing existing vectors, delete them first
            if replace_existing:
                self.delete_vectors_by_pdf_name(pdf_name)
                
            doc = fitz.open(file_path)
            print(f"Processing {file_path} with {len(doc)} pages")
            
            # List to collect vectors for batch upsert
            vectors_batch = []
            batch_size = 50  # Process in batches to be efficient
            
            for page_num in range(len(doc)):
                # Extract text from the page
                page_text = self.extract_text_from_pdf_page(doc, page_num)
                
                if not page_text:
                    print(f"Warning: No text extracted from page {page_num + 1} of {file_path}")
                    continue

                # Create embedding for the page
                embedding = self.create_embedding(page_text)
                if not embedding:
                    print(f"Warning: Failed to create embedding for page {page_num + 1} of {file_path}")
                    continue

                # Generate a sanitized base ID from PDF name
                sanitized_pdf_name = self.sanitize_vector_id(pdf_name)
                
                # Generate a unique 5-character ID for each page
                unique_id = self.generate_unique_id()
                
                # Combine into final vector ID (sanitized and guaranteed unique)
                vector_id = f"{sanitized_pdf_name}_{page_num}_{unique_id}"
                
                # Double-check the vector_id is valid (in case sanitization needs improvement)
                vector_id = self.sanitize_vector_id(vector_id)

                # Create metadata
                metadata = {
                    "file_name": os.path.basename(file_path),
                    "pdf_name": pdf_name,
                    "house_name": house_name,
                    "page_number": page_num + 1,
                    "text": page_text[:8000],  # Truncate for metadata
                    "char_count": len(page_text)
                }

                # Add to batch
                vectors_batch.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })
                
                # When batch is full or at the end, upload to Pinecone
                if len(vectors_batch) >= batch_size or page_num == len(doc) - 1:
                    if vectors_batch:
                        self.index.upsert(vectors=vectors_batch)
                        print(f"Uploaded batch of {len(vectors_batch)} vectors")
                        vectors_batch = []  # Reset batch

            doc.close()
            print(f"Successfully processed PDF: {pdf_name}")
            
        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
            raise  # Re-raise to see full stack trace when debugging

    def query_similar(self, query_text: str, top_k: int = 5):
        """Query Pinecone for similar documents based on the input text."""
        query_embedding = self.create_embedding(query_text)
        if not query_embedding:
            print("Failed to create query embedding")
            return {"matches": []}
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
        )
        return results

    def process_all_pdfs_in_folder(self, replace_existing: bool = True):
        """
        Process all PDFs in the given folder.
        If replace_existing is True, it will replace any existing vectors for PDFs with the same name.
        """
        for filename in os.listdir(self.folder_path):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(self.folder_path, filename)
                print(f"Processing {file_path}")
                self.embed_and_store_pdf(file_path, replace_existing=replace_existing)

    def process_specific_pdf(self, filename: str, house_name: str, replace_existing: bool = True):
        """
        Process a specific PDF file from the folder.
        If replace_existing is True, it will replace any existing vectors for the PDF with the same name.
        """
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(self.folder_path, filename)
            if os.path.exists(file_path):
                print(f"Processing specific file: {file_path}")
                self.embed_and_store_pdf(file_path, house_name = house_name, replace_existing=replace_existing)
            else:
                print(f"File not found: {file_path}")
        else:
            print(f"Not a PDF file: {filename}")


# Example usage
if __name__ == "__main__":
    vectorizer = WordVectorizerOpenAIPinecone(
        folder_path="D:\\brindyjean\\House Notes PDF",  # Specify the folder containing your PDFs
        pinecone_index_name="brindy-house-test-knowladgebase"
    )
    
    # Example 1: Process all PDFs in the folder (replacing existing ones with the same name)
    # vectorizer.process_all_pdfs_in_folder(replace_existing=True)
    
    # Example 2: Process a specific PDF file (replacing if it exists)
    vectorizer.process_specific_pdf("HN_ Newport Beach 1, 2, 3.pdf",
                                     house_name = "Newport Beach 1, 2, 3",
                                     replace_existing=True)