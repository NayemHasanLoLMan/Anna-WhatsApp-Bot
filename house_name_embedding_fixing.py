import random
import string
import os
import base64
from typing import List, Dict
import openai
from PIL import Image
import io
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from docx import Document
from docx.document import Document as DocumentType
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph
import re

load_dotenv()

class DocxVectorizerOpenAIPinecone:
    def __init__(
        self,
        folder_path: str,
        pinecone_index_name: str,
        chunk_size: int = 2000,
        chunk_overlap: int = 500
    ):
        """
        Initialize the DOCX vectorizer with OpenAI embeddings and Pinecone storage.
        Processes DOCX files from the specified folder path with improved text extraction.
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found at: {folder_path}")
        self.folder_path = folder_path
        self.index_name = pinecone_index_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # OpenAI setup - use ada-002 for compatibility with older OpenAI versions
        self.embedding_model = "text-embedding-ada-002"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Check OpenAI version to handle API differences
        try:
            # Try to import the new client (version 1.0+)
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.use_new_api = True
        except ImportError:
            # Fall back to old API
            self.client = None
            self.use_new_api = False

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
        """Generate a 8-character unique alphanumeric ID."""
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

    def sanitize_vector_id(self, vector_id: str) -> str:
        """Sanitize the vector ID to ensure it contains only ASCII characters and is valid for Pinecone."""
        # Replace spaces, parentheses, and other special characters with underscores
        sanitized = re.sub(r'[^\w\d]', '_', vector_id)
        # Ensure it's ASCII only by replacing any remaining non-ASCII characters
        sanitized = re.sub(r'[^\x00-\x7F]+', '_', sanitized)
        # Truncate if too long (Pinecone has limits on ID length)
        if len(sanitized) > 50:
            prefix = sanitized[:40]
            suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            sanitized = f"{prefix}_{suffix}"
        return sanitized

    def extract_image_from_docx_relationship(self, doc: Document, image_rel_id: str) -> Image.Image:
        """Extract image from DOCX using relationship ID."""
        try:
            # Get the image part from the document
            image_part = doc.part.related_parts[image_rel_id]
            image_bytes = image_part.blob
            image = Image.open(io.BytesIO(image_bytes))
            return image
        except Exception as e:
            print(f"Error extracting image with relationship ID {image_rel_id}: {e}")
            return None

    def extract_text_from_image_with_gpt4v(self, image: Image.Image) -> str:
        """Extract text from image using OpenAI GPT-4 Vision."""
        try:
            # Convert image to base64
            buffer = io.BytesIO()
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(buffer, format='JPEG', quality=85)
            image_bytes = buffer.getvalue()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')

            # Use OpenAI GPT-4 Vision to extract text
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Updated model name
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all text content from this image. If there are tables, preserve the structure. If there are charts or diagrams, describe the key information. Return only the extracted text without any additional commentary."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            extracted_text = response.choices[0].message.content.strip()
            return extracted_text if extracted_text else ""
            
        except Exception as e:
            print(f"Error extracting text from image using GPT-4V: {e}")
            return ""

    def extract_images_from_docx(self, doc: Document) -> List[str]:
        """Extract all images from DOCX and convert to text using GPT-4V."""
        image_texts = []
        
        # Skip image processing if we don't have vision API access
        if not self.use_new_api:
            print("Skipping image extraction - Vision API requires newer OpenAI library")
            return image_texts
        
        try:
            # Get all image relationships
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    try:
                        image_part = rel.target_part
                        image_bytes = image_part.blob
                        image = Image.open(io.BytesIO(image_bytes))
                        
                        # Extract text from image using GPT-4V
                        image_text = self.extract_text_from_image_with_gpt4v(image)
                        if image_text:
                            image_texts.append(f"[IMAGE CONTENT]: {image_text}")
                            
                    except Exception as e:
                        print(f"Error processing image: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error extracting images from document: {e}")
            
        return image_texts

    def extract_table_text(self, table: Table) -> str:
        """Extract text from a table with proper formatting."""
        table_text = []
        
        try:
            for row_idx, row in enumerate(table.rows):
                row_cells = []
                for cell in row.cells:
                    # Get all text from cell, including from nested paragraphs
                    cell_text = ""
                    for paragraph in cell.paragraphs:
                        if paragraph.text.strip():
                            cell_text += paragraph.text.strip() + " "
                    row_cells.append(cell_text.strip())
                
                if any(cell.strip() for cell in row_cells):  # Only add non-empty rows
                    # Use | as delimiter for better readability
                    table_text.append(" | ".join(row_cells))
            
            return "\n".join(table_text)
            
        except Exception as e:
            print(f"Error extracting table text: {e}")
            return ""

    def extract_comprehensive_text_from_docx(self, file_path: str) -> str:
        """Extract comprehensive text from DOCX including paragraphs, tables, headers, footers, and images."""
        try:
            doc = Document(file_path)
            all_content = []
            
            # Extract text from document body in order
            for element in doc.element.body:
                if isinstance(element, CT_P):  # Paragraph
                    paragraph = Paragraph(element, doc)
                    if paragraph.text.strip():
                        all_content.append(paragraph.text.strip())
                        
                elif isinstance(element, CT_Tbl):  # Table
                    table = Table(element, doc)
                    table_text = self.extract_table_text(table)
                    if table_text:
                        all_content.append(f"[TABLE]\n{table_text}\n[/TABLE]")

            # Extract text from headers
            for section in doc.sections:
                if section.header:
                    header_text = ""
                    for paragraph in section.header.paragraphs:
                        if paragraph.text.strip():
                            header_text += paragraph.text.strip() + "\n"
                    if header_text:
                        all_content.insert(0, f"[HEADER]\n{header_text.strip()}\n[/HEADER]")
                
                # Extract text from footers
                if section.footer:
                    footer_text = ""
                    for paragraph in section.footer.paragraphs:
                        if paragraph.text.strip():
                            footer_text += paragraph.text.strip() + "\n"
                    if footer_text:
                        all_content.append(f"[FOOTER]\n{footer_text.strip()}\n[/FOOTER]")

            # Extract images and their text content
            image_texts = self.extract_images_from_docx(doc)
            all_content.extend(image_texts)
            
            # Join all content
            full_text = "\n\n".join(all_content)
            return full_text
            
        except Exception as e:
            print(f"Error extracting comprehensive text from DOCX {file_path}: {e}")
            return ""

    def chunk_text_by_character_count(self, text: str) -> List[str]:
        """Split text into chunks based on character count with overlap."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to break at a sentence or paragraph
            if end < len(text):
                # Look for sentence endings within the last 200 characters
                break_points = ['. ', '.\n', '!\n', '?\n', '\n\n']
                best_break = end
                
                for i in range(min(200, end - start)):
                    pos = end - i
                    for bp in break_points:
                        if text[pos:pos+len(bp)] == bp:
                            best_break = pos + len(bp)
                            break
                    if best_break != end:
                        break
                
                end = best_break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + 1, end - self.chunk_overlap)
            
            # Prevent infinite loop
            if start >= len(text):
                break
                
        return chunks

    def create_embedding(self, text: str) -> List[float]:
        """Create an embedding for the given text using OpenAI's embedding model."""
        try:
            # Truncate text to avoid exceeding token limits
            truncated = text[:8000]
            
            if self.use_new_api:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=truncated
                )
                return response.data[0].embedding
            else:
                # Old API format
                response = openai.Embedding.create(
                    model=self.embedding_model,
                    input=truncated
                )
                return response['data'][0]['embedding']
        except Exception as e:
            print(f"Error creating embedding: {e}")
            return []

    def delete_vectors_by_document_name(self, document_name: str):
        """Delete all existing vectors in the index related to a specific document name."""
        try:
            # Query to find all vectors for this document
            results = self.index.query(
                vector=[0.0] * 1536,
                top_k=10000,
                include_metadata=True,
                filter={"document_name": {"$eq": document_name}}
            )
            
            if results and "matches" in results:
                vector_ids = [match["id"] for match in results["matches"] if "id" in match]
                
                if vector_ids:
                    # Delete in batches
                    batch_size = 100
                    for i in range(0, len(vector_ids), batch_size):
                        batch = vector_ids[i:i+batch_size]
                        self.index.delete(ids=batch)
                    
                    print(f"Deleted {len(vector_ids)} vectors for document: {document_name}")
                else:
                    print(f"No vectors found for document: {document_name}")
            else:
                print(f"No matches found for document: {document_name}")
                
        except Exception as e:
            print(f"Error deleting vectors for document {document_name}: {e}")

    def embed_and_store_docx(self, file_path: str, house_name: str, replace_existing: bool = True):
        """
        Process a single DOCX document and create embeddings based on character count chunks.
        Store embeddings in Pinecone with comprehensive metadata.
        """
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        try:
            # If replacing existing vectors, delete them first
            if replace_existing:
                self.delete_vectors_by_document_name(file_name)
            
            print(f"Processing DOCX: {file_path}")
            
            # Extract comprehensive text from DOCX
            document_text = self.extract_comprehensive_text_from_docx(file_path)
            
            if not document_text.strip():
                print(f"Warning: No text extracted from {file_path}")
                return
            
            print(f"Extracted {len(document_text)} characters from document")
            
            # Split text into chunks based on character count
            text_chunks = self.chunk_text_by_character_count(document_text)
            print(f"Created {len(text_chunks)} chunks for processing")
            
            # Process chunks in batches
            vectors_batch = []
            batch_size = 50
            
            for chunk_idx, chunk_text in enumerate(text_chunks):
                # Create embedding for the chunk
                embedding = self.create_embedding(chunk_text)
                if not embedding:
                    print(f"Failed to create embedding for chunk {chunk_idx}")
                    continue

                # Generate vector ID
                sanitized_file_name = self.sanitize_vector_id(file_name)
                unique_id = self.generate_unique_id()
                vector_id = f"{sanitized_file_name}_chunk_{chunk_idx}_{unique_id}"
                vector_id = self.sanitize_vector_id(vector_id)

                # Create comprehensive metadata
                metadata = {
                    "file_name": os.path.basename(file_path),
                    "document_name": file_name,
                    "house_name": house_name,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(text_chunks),
                    "text": chunk_text[:8000],  # Store text in metadata for retrieval
                    "char_count": len(chunk_text),
                    "file_type": "docx",
                    "extraction_method": "comprehensive_with_gpt4v"
                }

                vectors_batch.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })
                
                # Upload batch when it reaches batch_size or is the last chunk
                if len(vectors_batch) >= batch_size or chunk_idx == len(text_chunks) - 1:
                    if vectors_batch:
                        self.index.upsert(vectors=vectors_batch)
                        print(f"Uploaded batch of {len(vectors_batch)} vectors (chunks {chunk_idx - len(vectors_batch) + 1} to {chunk_idx})")
                        vectors_batch = []
                        
            print(f"Successfully processed document: {file_name} with {len(text_chunks)} chunks")
            
        except Exception as e:
            print(f"Error processing document {file_path}: {e}")
            raise

    def query_similar(self, query_text: str, top_k: int = 5, house_name: str = None):
        """Query Pinecone for similar documents based on the input text."""
        query_embedding = self.create_embedding(query_text)
        if not query_embedding:
            print("Failed to create query embedding")
            return {"matches": []}
            
        # Add filter for house_name if provided
        filter_dict = {}
        if house_name:
            filter_dict["house_name"] = {"$eq": house_name}
            
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        return results

    def process_specific_docx(self, filename: str, house_name: str, replace_existing: bool = True):
        """
        Process a specific DOCX file from the folder.
        If replace_existing is True, it will replace any existing vectors for the document with the same name.
        """
        if not filename.lower().endswith('.docx'):
            print(f"File is not a DOCX file: {filename}")
            return
            
        file_path = os.path.join(self.folder_path, filename)
        if os.path.exists(file_path):
            print(f"Processing specific DOCX file: {file_path}")
            self.embed_and_store_docx(file_path, house_name=house_name, replace_existing=replace_existing)
        else:
            print(f"File not found: {file_path}")

    def get_document_stats(self, document_name: str = None):
        """Get statistics about stored documents."""
        try:
            # First, let's get all vectors without any filter to see what's there
            all_results = self.index.query(
                vector=[0.0] * 1536,
                top_k=10000,
                include_metadata=True
            )
            
            print(f"Total vectors in index: {len(all_results.get('matches', []))}")
            
            if document_name:
                # Filter results for specific document
                filtered_matches = [
                    match for match in all_results.get('matches', [])
                    if match.get('metadata', {}).get('document_name') == document_name
                ]
                
                if filtered_matches:
                    total_chunks = len(filtered_matches)
                    total_chars = sum(match["metadata"].get("char_count", 0) for match in filtered_matches)
                    
                    print(f"Statistics for document '{document_name}':")
                    print(f"  Total chunks: {total_chunks}")
                    print(f"  Total characters: {total_chars:,}")
                    
                    # Show sample metadata
                    if filtered_matches:
                        sample_meta = filtered_matches[0]['metadata']
                        print(f"  Sample metadata keys: {list(sample_meta.keys())}")
                else:
                    print(f"No vectors found for document: {document_name}")
                    # Show available document names
                    doc_names = set(
                        match.get('metadata', {}).get('document_name', 'Unknown')
                        for match in all_results.get('matches', [])
                    )
                    print(f"Available documents: {doc_names}")
            else:
                # Show stats for all documents
                if all_results.get('matches'):
                    total_chunks = len(all_results["matches"])
                    total_chars = sum(match["metadata"].get("char_count", 0) for match in all_results["matches"])
                    documents = set(match["metadata"].get("document_name", "") for match in all_results["matches"])
                    
                    print(f"Statistics:")
                    print(f"  Total chunks: {total_chunks}")
                    print(f"  Total characters: {total_chars:,}")
                    print(f"  Documents: {len(documents)}")
                    
                    for doc in sorted(documents):
                        doc_chunks = sum(1 for match in all_results["matches"] if match["metadata"].get("document_name") == doc)
                        print(f"    {doc}: {doc_chunks} chunks")
                else:
                    print("No vectors found in the index")
                    
        except Exception as e:
            print(f"Error getting document stats: {e}")
            import traceback
            traceback.print_exc()


# Example usage
if __name__ == "__main__":
    vectorizer = DocxVectorizerOpenAIPinecone(
        folder_path="C:\\Users\\hasan\\Downloads\\Hasan Files\\Anna-WhatsApp-Bot\\House Notes (Internal)",
        pinecone_index_name="houseinformation-embeddings",
        chunk_size=2000,  # Characters per chunk
        chunk_overlap=500  # Character overlap between chunks
    )
   
    # Process a specific DOCX file
    vectorizer.process_specific_docx(
        "HN_ 40th Pl.docx",  # Make sure this is a .docx file
        house_name="40th Pl",
        replace_existing=True
    )
    
    # Get statistics about the processed document
    vectorizer.get_document_stats("HN_ 40th Pl")
    
