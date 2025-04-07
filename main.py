import os
from pinecone import Pinecone
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Tuple
import time
import re
from tenacity import retry, stop_after_attempt, wait_exponential

from dotenv import load_dotenv


class HouseNotesBot:
    def __init__(self):
        # API keys
#         # API keys from environment variables
        self.PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
        self.GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
        
        
        # Index names
        self.HOUSE_INDEX_NAME = "house-notes-doc-embeddings"
        self.GUEST_GUIDE_INDEX_NAME = "guest-messaging-guide-doc-embeddings"
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.PINECONE_API_KEY)
        self.house_index = self.pc.Index(self.HOUSE_INDEX_NAME)
        self.guest_guide_index = self.pc.Index(self.GUEST_GUIDE_INDEX_NAME)
        
        # Initialize Gemini
        genai.configure(api_key=self.GOOGLE_API_KEY)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Bot state
        self.mode = "general"  # Modes: "general", "house_info", "guest_guide"
        self.current_house = None
        
        # Conversation memory
        self.chat_history = []
        
        # House information cache
        self.house_cache = {}
        self.load_house_names()
        
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text using Gemini"""
        embedding_model = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return embedding_model['embedding']
    
    def update_chat_history(self, role: str, content: str):
        """Add message to chat history with house context"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.chat_history.append({
            "role": role,
            "content": content,
            "timestamp": timestamp,
            "mode": self.mode,
            "house": self.current_house if self.mode == "house_info" else None
        })
        
        # Keep only recent history (last 10 messages)
        if len(self.chat_history) > 5:
            self.chat_history = self.chat_history[-5:]
    
    def detect_house_name(self, query: str) -> Optional[str]:
        """Enhanced house name detection with fuzzy matching"""
        # First, get all house names if cache is empty
        if not self.house_cache:
            try:
                # Query index with a dummy vector to get metadata
                dummy_vector = [0.0] * 768
                results = self.house_index.query(vector=dummy_vector, top_k=100, include_metadata=True)
                
                # Extract unique house names
                for match in results.get('matches', []):
                    house_name = match['metadata'].get('house_name')
                    if house_name and house_name != "Unknown":
                        # Store standard version
                        self.house_cache[house_name.lower()] = house_name
                        
                        # Also store without spaces
                        no_space = house_name.lower().replace(" ", "")
                        self.house_cache[no_space] = house_name
                        
                        # Store without special characters
                        simple_name = re.sub(r'[^a-zA-Z0-9]', '', house_name.lower())
                        self.house_cache[simple_name] = house_name
            except Exception as e:
                print(f"Error fetching house names: {e}")
        
        # Check if query contains any house name
        query_lower = query.lower().strip()
        query_no_space = query_lower.replace(" ", "")
        
        # Try exact matches first
        for house_key, house_name in self.house_cache.items():
            if house_key == query_lower or house_key in query_lower:
                return house_name
        
        # Try fuzzy matching for similar names
        for house_key, house_name in self.house_cache.items():
            # Check if query without spaces matches house key without spaces
            if house_key.replace(" ", "") == query_no_space:
                return house_name
            
            # Check for significant substring matches (at least 70% of the house name)
            if len(house_key) > 4 and (house_key in query_lower or query_lower in house_key):
                return house_name
        
        # If no match and we have a current house, maintain it unless explicitly changing
        if self.current_house:
            change_indicators = ["switch to", "change to", "use house", "different house", "another house"]
            if any(indicator in query_lower for indicator in change_indicators):
                return None  # User wants to change houses but no valid house detected
            return self.current_house
                
        return None
    
    def query_vector_db(self, query: str, index, filter_dict: Optional[Dict] = None, top_k: int = 8) -> List[Dict[str, Any]]:
        """Query specified Pinecone index with filtering options"""
        query_vector = self.embed_text(query)
        
        try:
            results = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            return results.get('matches', [])
        except Exception as e:
            print(f"Error querying index: {e}")
            return []
    
    def query_house_info(self, query: str, house_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query house information with optional house name filter"""
        # Try with house filter first if provided
        if house_name:
            filter_dict = {"house_name": {"$eq": house_name}}
            results = self.query_vector_db(query, self.house_index, filter_dict)
            
            # If no results with filter, try without filter
            if not results:
                # Log that we're trying without filter
                print(f"No results for {house_name}, trying general search")
                results = self.query_vector_db(query, self.house_index, None)
                
            return results
        else:
            # No house specified, try general search
            return self.query_vector_db(query, self.house_index, None)
    
    def query_guest_guide(self, query: str, top_k: int = 12) -> List[Dict[str, Any]]:
        """Query guest messaging guide with increased result count"""
        return self.query_vector_db(query, self.guest_guide_index, top_k=top_k)
    
    def format_chat_history(self) -> str:
        """Format chat history for context"""
        formatted = []
        current_mode_history = [msg for msg in self.chat_history if msg["mode"] == self.mode]
        
        for msg in current_mode_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted)
    
    def format_house_matches(self, matches: List[Dict]) -> Tuple[str, List[str]]:
        """Format house info matches into context string and source references"""
        context_chunks = []
        sources = []
        
        for match in matches:
            document_name = match['metadata'].get('file_name', 'Unknown document')
            text = match['metadata'].get('text', '')
            house_name = match['metadata'].get('house_name', 'Unknown house')
            
            if text:
                context_chunks.append(f"[Source: {document_name}, House: {house_name}]\n{text}")
                if document_name not in sources:
                    sources.append(f"{document_name} (House: {house_name})")
        
        return "\n\n".join(context_chunks), sources
    
    def format_guest_guide_matches(self, matches: List[Dict]) -> str:
        """Format guest guide matches into comprehensive context string"""
        # Extract and deduplicate text chunks by content
        unique_chunks = {}
        section_titles = set()
        
        for match in matches:
            text = match['metadata'].get('text', '').strip()
            if not text:
                continue
                
            # Extract section titles if available
            lines = text.split('\n')
            potential_title = lines[0] if lines else ""
            if potential_title and len(potential_title) < 100 and potential_title.endswith(':'):
                section_titles.add(potential_title)
            
            # Use first 100 chars as key to deduplicate similar chunks
            key = text[:100]
            if key not in unique_chunks:
                unique_chunks[key] = {
                    'text': text,
                    'score': match['score']  # Relevance score from vector DB
                }
            elif match['score'] > unique_chunks[key]['score']:
                # Update if this match has higher relevance
                unique_chunks[key]['score'] = match['score']
        
        # Sort chunks by relevance score
        sorted_chunks = sorted(
            unique_chunks.values(), 
            key=lambda x: x['score'], 
            reverse=True
        )
        
        # Format with section headers if available
        formatted_text = ""
        if section_titles:
            formatted_text += "## GUEST MESSAGING GUIDE SECTIONS:\n"
            formatted_text += "\n".join(f"- {title}" for title in sorted(section_titles))
            formatted_text += "\n\n"
        
        formatted_text += "## DETAILED GUIDELINES:\n\n"
        for chunk in sorted_chunks:
            formatted_text += f"{chunk['text']}\n\n"
            
        return formatted_text
    
    def generate_general_response(self, query: str) -> str:
        """Generate response in general AI mode"""
        prompt = f"""You are a helpful AI assistant that can answer general questions.
        
                Recent conversation:
                {self.format_chat_history()}

                User question: {query}

                Response:"""

        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"I'm sorry, I encountered an error in general mode: {str(e)}"
    
    def generate_house_info_response(self, query: str) -> str:
        """Generate response about house information with improved house name handling"""
        # First check if this is an explicit house request pattern
        explicit_house_request = re.search(r'(information|details|about|for|at|in)\s+(?:the\s+)?([a-zA-Z\s]+)$', query, re.IGNORECASE)
        
        # Detect house name in query
        detected_house = None
        if explicit_house_request:
            # Extract the potential house name from the end of the query
            potential_house = explicit_house_request.group(2).strip()
            # Try to match this potential house name
            for house_key, house_name in self.house_cache.items():
                if house_key == potential_house.lower() or house_name.lower() == potential_house.lower():
                    detected_house = house_name
                    break
        
        # If no explicit house found, try general detection
        if not detected_house:
            detected_house = self.detect_house_name(query)
        
        # Handle house transitions
        house_switch_msg = ""
        if detected_house:
            if self.current_house != detected_house:
                previous_house = self.current_house
                self.current_house = detected_house
                house_switch_msg = f"Now discussing {self.current_house}."
        
        # Get relevant documents
        matches = []
        if self.current_house:
            # Try with current house first
            matches = self.query_house_info(query, self.current_house)
        
        # If no matches or no current house, try general search
        if not matches:
            matches = self.query_house_info(query)
            
            # If we found matches but don't have a current house, check if they're all for the same house
            if matches and not self.current_house:
                house_counts = {}
                for match in matches:
                    house = match['metadata'].get('house_name')
                    if house:
                        house_counts[house] = house_counts.get(house, 0) + 1
                
                # If all or most results are for one house, set it as current
                if house_counts:
                    most_common_house = max(house_counts.items(), key=lambda x: x[1])[0]
                    if house_counts[most_common_house] >= len(matches) * 0.7:  # If 70% or more matches are for one house
                        self.current_house = most_common_house
                        house_switch_msg = f"Now discussing {self.current_house}."
        
        # Format context from matches
        context, sources = self.format_house_matches(matches)
        
        if not context:
            houses = list(set(self.house_cache.values()))
            houses_sample = ", ".join(houses[:5])
            if len(houses) > 5:
                houses_sample += f", and {len(houses) - 5} more"
                
            if self.current_house:
                return f"I don't have specific information about that for {self.current_house}. Please try asking something else about the property."
            else:
                return f"I don't have information about that. Please specify a house name like: {houses_sample}"
        
        # Create prompt for Gemini
        house_context = f"Current house being discussed: {self.current_house}" if self.current_house else "No specific house selected"
        prompt = f"""You are a helpful assistant for answering questions about rental properties and houses.
                
                {house_context}

                Recent conversation:
                {self.format_chat_history()}

                Use the following information to answer the latest question about house properties. Be conversational and friendly.
                If you know which specific house the question is about, mention the house name in your response.
                Include specific details when available but don't make up information.

                Context information:
                {context}

                User question: {query}

                Response:"""

        try:
            response = self.gemini_model.generate_content(prompt)
            answer = response.text
            
            # Add house switch message and sources if needed
            if house_switch_msg:
                answer = f"{house_switch_msg}\n\n{answer}"
                    
            # Add sources info if we have matches
            if sources:
                answer += f"\n\n(Information sourced from: {', '.join(sources)})"
                    
            return answer
        except Exception as e:
            return f"I'm sorry, I encountered an error in house info mode: {str(e)}"
        


    def load_house_names(self):
        """Pre-load house names at startup"""
        try:
            print("Loading house names...")
            # Query index with a dummy vector to get metadata
            dummy_vector = [0.0] * 768
            results = self.house_index.query(vector=dummy_vector, top_k=100, include_metadata=True)
            
            # Extract unique house names
            houses_found = 0
            for match in results.get('matches', []):
                house_name = match['metadata'].get('house_name')
                if house_name and house_name != "Unknown" and house_name.lower() not in self.house_cache:
                    self.house_cache[house_name.lower()] = house_name
                    houses_found += 1
                    
                    # Also store without spaces and special characters
                    no_space = house_name.lower().replace(" ", "")
                    self.house_cache[no_space] = house_name
                    
                    simple_name = re.sub(r'[^a-zA-Z0-9]', '', house_name.lower())
                    if simple_name != house_name.lower():
                        self.house_cache[simple_name] = house_name
            
            print(f"Loaded {houses_found} unique houses into cache")
            return houses_found > 0
        except Exception as e:
            print(f"Error loading house names: {e}")
            return False
     
    
    def generate_guest_guide_response(self, query: str) -> str:
        """Generate response based on guest messaging guide"""
        # Get more relevant documents from guest guide for comprehensive coverage
        matches = self.query_guest_guide(query, top_k=15)
        
        if not matches:
            return "I don't have specific guest messaging guidelines for that query. Please try asking something else or switch to another mode."
        
        # Format context from matches - specialized for guest guide
        context = self.format_guest_guide_matches(matches)
        
        # Create prompt for Gemini with specific instructions for guest guide responses
        prompt = f"""You are a professional host assistant specializing in guest messaging. Your task is to provide comprehensive responses based on the guest messaging guide.

            Recent conversation:
            {self.format_chat_history()}

            IMPORTANT INSTRUCTIONS:
            1. Start with the EXACT template that best matches the query
            2. After providing the exact template, include any additional relevant templates or guidelines that may be helpful
            3. For each template used, maintain the EXACT wording - do not paraphrase
            4. Fill in ANY placeholder values (like $00.00) with appropriate values if specified
            5. Add helpful context or related information from the guide that could be relevant
            6. Format your response in clear sections:
            - Primary Response (exact template)
            - Related Templates (if any)
            - Additional Guidelines (if relevant)
            - Contextual Notes (AI-generated insights based on the guide's tone and style)

            Guest messaging guide content:
            {context}

            User question: {query}

            Response (using EXACT wording from the guide when available):"""

        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"I'm sorry, I encountered an error in guest guide mode: {str(e)}"
    
    def process_command(self, command: str) -> str:
        """Process special commands"""
        command = command.lower().strip()
        
        if command == "/general":
            self.mode = "general"
            return "Switched to general AI chat mode. I'll respond to your questions using my general knowledge."
            
        elif command == "/house_info":
            self.mode = "house_info"
            houses = list(self.house_cache.values()) if self.house_cache else []
            house_list = ", ".join(houses) if houses else "No houses found in database"
            return f"Switched to house information mode. I'll answer questions about properties.\nAvailable houses: {house_list}"
            
        elif command == "/guest_guide":
            self.mode = "guest_guide"
            return "Switched to guest messaging guide mode. I'll provide responses based on the guest messaging guidelines."
            
        elif command.startswith("/use_house:"):
            if self.mode != "house_info":
                self.mode = "house_info"
                mode_msg = "Switched to house information mode. "
            else:
                mode_msg = ""
                
            house_name = command[11:].strip()
            lower_house = house_name.lower()
            
            if lower_house in [h.lower() for h in self.house_cache.values()]:
                for k, v in self.house_cache.items():
                    if v.lower() == lower_house:
                        self.current_house = v
                        return f"{mode_msg}Now using {v} as the current house."
            else:
                return f"{mode_msg}I don't recognize '{house_name}'. Please specify a valid house name."
                
        elif command == "/help":
            return """
                Available commands:
                - /general - Switch to general AI chat mode
                - /house_info - Switch to house information mode
                - /guest_guide - Switch to guest messaging guide mode
                - /use_house:[house name] - Set the current house
                - /help - Show this help message
                - /exit - End the conversation
                """
        else:
            return f"Unknown command: {command}. Type /help for available commands."
    
    def generate_response(self, user_input: str) -> str:
        """Generate response based on current mode"""
        # Check if input is a command
        if user_input.startswith("/"):
            return self.process_command(user_input)
        
        # Generate response based on current mode
        if self.mode == "general":
            return self.generate_general_response(user_input)
        elif self.mode == "house_info":
            return self.generate_house_info_response(user_input)
        elif self.mode == "guest_guide":
            return self.generate_guest_guide_response(user_input)
        else:
            self.mode = "general"  # Reset to general mode if unknown
            return self.generate_general_response(user_input)
    
    def chat(self):
        """Interactive chat loop"""
        print("House Notes Bot: Hello! I can operate in different modes:\n"
              "- General AI chat (default)\n"
              "- House information (/house_info)\n" 
              "- Guest messaging guide (/guest_guide)\n"
              "Type /help for all commands or /exit to end the conversation.")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == '/exit':
                print("House Notes Bot: Goodbye!")
                break
                
            # Update chat history
            self.update_chat_history("user", user_input)
            
            # Generate and display response
            start_time = time.time()
            response = self.generate_response(user_input)
            end_time = time.time()
            
            print(f"\nHouse Notes Bot ({self.mode.replace('_', ' ')} mode): {response}")
            print(f"[Response time: {end_time - start_time:.2f}s]")
            
            # Update chat history with bot response
            self.update_chat_history("assistant", response)


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    try:
        bot = HouseNotesBot()
        bot.chat()
    except Exception as e:
        print(f"Error initializing bot: {str(e)}")



# import os
# from pinecone import Pinecone
# import google.generativeai as genai
# from typing import List, Dict, Any, Optional, Tuple
# import time
# import re
# from tenacity import retry, stop_after_attempt, wait_exponential

# from dotenv import load_dotenv

# class HouseNotesBot:
#     def __init__(self):
#         """Initialize the HouseNotesBot with environment variables and configurations."""
#         # API keys from environment variables
#         self.PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
#         self.GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
        
#         # Index names
#         self.HOUSE_INDEX_NAME = "house-notes-doc-embeddings"
#         self.GUEST_GUIDE_INDEX_NAME = "guest-messaging-guide-doc-embeddings"
        
#         # Initialize Pinecone
#         self.pc = Pinecone(api_key=self.PINECONE_API_KEY)
#         self.house_index = self.pc.Index(self.HOUSE_INDEX_NAME)
#         self.guest_guide_index = self.pc.Index(self.GUEST_GUIDE_INDEX_NAME)
        
#         # Set default embedding dimension for Pinecone (standard for most models)
#         self.embedding_dim = 768  # Standard dimension for many embedding models
        
#         # Initialize Gemini
#         genai.configure(api_key=self.GOOGLE_API_KEY)
#         self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        
#         # Bot state
#         self.mode: str = "general"
#         self.current_house: Optional[str] = None
        
#         # Conversation memory
#         self.chat_history: List[Dict[str, str]] = []
        
#         # House information cache with TTL
#         self.house_cache: Dict[str, Tuple[str, float]] = {}  # {house_key: (house_name, timestamp)}
#         self.cache_ttl: float = 3600  # 1 hour TTL
        
#     @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
#     def embed_text(self, text: str) -> List[float]:
#         """Generate embeddings for text using Gemini with retry logic.
        
#         Args:
#             text: Input text to embed
            
#         Returns:
#             List of float values representing the embedding
#         """
#         embedding_model = genai.embed_content(
#             model="models/embedding-001",
#             content=text,
#             task_type="retrieval_document"
#         )
#         return embedding_model['embedding']
    
#     def update_chat_history(self, role: str, content: str) -> None:
#         """Add message to chat history with timestamp and mode."""
#         timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
#         self.chat_history.append({
#             "role": role,
#             "content": content,
#             "timestamp": timestamp,
#             "mode": self.mode
#         })
#         if len(self.chat_history) > 10:
#             self.chat_history = self.chat_history[-10:]
    
#     def detect_house_name(self, query: str) -> Optional[str]:
#         """Detect house name from query using cached house names."""
#         current_time = time.time()
        
#         # Refresh cache if empty or expired
#         if not self.house_cache or all(t < current_time - self.cache_ttl for _, t in self.house_cache.values()):
#             try:
#                 dummy_vector = [0.0] * self.embedding_dim
#                 results = self.house_index.query(vector=dummy_vector, top_k=100, include_metadata=True)
#                 self.house_cache.clear()
#                 for match in results.get('matches', []):
#                     house_name = match['metadata'].get('house_name')
#                     if house_name and house_name != "Unknown":
#                         self.house_cache[house_name.lower()] = (house_name, current_time)
#             except Exception as e:
#                 print(f"Error fetching house names: {e}")
        
#         query_lower = query.lower()
#         for house_key, (house_name, _) in self.house_cache.items():
#             if house_key in query_lower:
#                 return house_name
#         return self.current_house
    
#     def query_vector_db(self, query: str, index, filter_dict: Optional[Dict] = None, top_k: int = 8) -> List[Dict[str, Any]]:
#         """Query Pinecone index with retry logic."""
#         query_vector = self.embed_text(query)
#         try:
#             results = index.query(
#                 vector=query_vector,
#                 top_k=top_k,
#                 include_metadata=True,
#                 filter=filter_dict
#             )
#             return results.get('matches', [])
#         except Exception as e:
#             print(f"Error querying index: {e}")
#             return []
    
#     def query_house_info(self, query: str, house_name: Optional[str] = None) -> List[Dict[str, Any]]:
#         """Query house information with optional house name filter."""
#         filter_dict = {"house_name": {"$eq": house_name}} if house_name else None
#         return self.query_vector_db(query, self.house_index, filter_dict)
    
#     def query_guest_guide(self, query: str, top_k: int = 12) -> List[Dict[str, Any]]:
#         """Query guest messaging guide."""
#         return self.query_vector_db(query, self.guest_guide_index, top_k=top_k)
    
#     def format_chat_history(self) -> str:
#         """Format chat history for context."""
#         formatted = [f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
#                     for msg in self.chat_history if msg["mode"] == self.mode]
#         return "\n".join(formatted)
    
#     def format_house_matches(self, matches: List[Dict]) -> Tuple[str, List[str]]:
#         """Format house info matches into context string and source references."""
#         context_chunks, sources = [], []
#         for match in matches:
#             document_name = match['metadata'].get('file_name', 'Unknown document')
#             text = match['metadata'].get('text', '')
#             house_name = match['metadata'].get('house_name', 'Unknown house')
#             if text:
#                 context_chunks.append(f"[Source: {document_name}, House: {house_name}]\n{text}")
#                 if document_name not in sources:
#                     sources.append(f"{document_name} (House: {house_name})")
#         return "\n\n".join(context_chunks), sources
    
#     def format_guest_guide_matches(self, matches: List[Dict], query: str) -> Tuple[str, str, List[str]]:
#         """Format guest guide matches into primary template, additional context, and sources."""
#         primary_template, additional_context, sources = "", "", []
#         unique_chunks = {}
        
#         for match in matches:
#             text = match['metadata'].get('text', '').strip()
#             if not text:
#                 continue
#             document_name = match['metadata'].get('file_name', 'Unknown document')
#             key = text[:100]
#             if key not in unique_chunks or match['score'] > unique_chunks[key]['score']:
#                 unique_chunks[key] = {'text': text, 'score': match['score'], 'source': document_name}
        
#         sorted_chunks = sorted(unique_chunks.values(), key=lambda x: x['score'], reverse=True)
#         if sorted_chunks:
#             # Use highest scoring match as primary template if it matches the query intent
#             for chunk in sorted_chunks:
#                 if query.lower() in chunk['text'].lower() or any(keyword in chunk['text'].lower() for keyword in query.lower().split()):
#                     primary_template = chunk['text']
#                     sources.append(chunk['source'])
#                     break
#             if not primary_template:  # Fallback to highest scoring if no direct match
#                 primary_template = sorted_chunks[0]['text']
#                 sources.append(sorted_chunks[0]['source'])
            
#             # Additional context only if relevant to query
#             additional_chunks = [chunk['text'] for chunk in sorted_chunks if chunk['text'] != primary_template and 
#                                 (query.lower() in chunk['text'].lower() or any(keyword in chunk['text'].lower() for keyword in query.lower().split()))]
#             if additional_chunks:
#                 additional_context = "\n\n".join(additional_chunks)
#                 sources.extend(chunk['source'] for chunk in sorted_chunks if chunk['text'] in additional_chunks and chunk['source'] not in sources)
        
#         return primary_template, additional_context, sources
    
#     def generate_general_response(self, query: str) -> str:
#         """Generate response in general AI mode."""
#         prompt = f"""You are a helpful AI assistant.\n\nRecent conversation:\n{self.format_chat_history()}\n\nUser question: {query}\n\nResponse:"""
#         try:
#             response = self.gemini_model.generate_content(prompt)
#             return response.text
#         except Exception as e:
#             return f"Error in general mode: {str(e)}"
    
#     def generate_house_info_response(self, query: str) -> str:
#         """Generate response about house information specific to the detected house."""
#         detected_house = self.detect_house_name(query)
#         if detected_house and detected_house != self.current_house:
#             self.current_house = detected_house
#             house_switch_msg = f"Now discussing {self.current_house}.\n\n"
#         else:
#             house_switch_msg = "" if not self.current_house else f"Discussing {self.current_house}.\n\n"
        
#         if not self.current_house:
#             return f"{house_switch_msg}Please specify a house name (e.g., 'Mesa Coastal') to get accurate information."
        
#         # Query only for the specific house
#         matches = self.query_house_info(query, self.current_house)
#         context, sources = self.format_house_matches(matches)
        
#         if not context:
#             return f"{house_switch_msg}I don't have specific information about that for {self.current_house}. Please try a different question or check if the house name is correct."
        
#         prompt = f"""You are a helpful assistant for rental properties.
        
#     Current house: {self.current_house}

#     Recent conversation:
#     {self.format_chat_history()}

#     Context information (specific to {self.current_house}):
#     {context}

#     User question: {query}

#     Provide a detailed, conversational response using only the information relevant to {self.current_house}. Include all necessary details from the context that answer the query. Do not invent information if it's not in the context.
#     Response:"""
        
#         try:
#             response = self.gemini_model.generate_content(prompt)
#             return f"{house_switch_msg}{response.text}\n\n(Information sourced from: {', '.join(sources)})"
#         except Exception as e:
#             return f"{house_switch_msg}Error in house info mode: {str(e)}"
    
#     def generate_guest_guide_response(self, query: str) -> str:
#         """Generate response based on guest messaging guide with exact primary template."""
#         matches = self.query_guest_guide(query, top_k=15)
#         if not matches:
#             return "I don't have specific guest messaging guidelines for that query."
        
#         primary_template, additional_context, sources = self.format_guest_guide_matches(matches, query)
#         if not primary_template:
#             return "No exact template found for your query in the guest messaging guide."
        
#         # Construct prompt to refine the primary response while keeping exact wording
#         prompt = f"""You are a professional host assistant specializing in guest messaging.

#     Guest messaging guide primary template (use EXACT wording):
#     {primary_template}

#     Additional relevant guidelines:
#     {additional_context}

#     User query: {query}

#     Instructions:
#     1. Start with the EXACT primary template text provided above.
#     2. Fill in any placeholders (e.g., $00.00, [CHECK-IN / CHECK-OUT]) with appropriate values based on the query or context if specified; otherwise, leave them as is.
#     3. After the exact template, optionally add brief, relevant clarification or additional info from the additional guidelines if it enhances the response, but keep it concise.
#     4. Maintain a friendly, professional tone consistent with the guide.

#     Response:"""
        
#         try:
#             response = self.gemini_model.generate_content(prompt).text
#             return f"{response}\n\n(Sourced from: {', '.join(sources)})"
#         except Exception as e:
#             return f"Error in guest guide mode: {str(e)}\n\n(Sourced from: {', '.join(sources)})"
#         return response
    
#     def process_command(self, command: str) -> str:
#         """Process special commands with input validation."""
#         command = re.sub(r'[^\w\s:/-]', '', command.lower().strip())  # Basic sanitization
        
#         if command == "/general":
#             self.mode = "general"
#             return "Switched to general AI chat mode."
#         elif command == "/house_info":
#             self.mode = "house_info"
#             houses = [name for name, _ in self.house_cache.values()]
#             return f"Switched to house information mode.\nAvailable houses: {', '.join(houses) or 'None'}"
#         elif command == "/guest_guide":
#             self.mode = "guest_guide"
#             return "Switched to guest messaging guide mode."
#         elif command.startswith("/use_house:"):
#             house_name = command[11:].strip()
#             if not house_name:
#                 return "Please specify a house name after /use_house:"
#             self.mode = "house_info"
#             lower_house = house_name.lower()
#             for k, (v, _) in self.house_cache.items():
#                 if k == lower_house:
#                     self.current_house = v
#                     return f"Now using {v} as the current house."
#             return f"I don't recognize '{house_name}'. Please specify a valid house name."
#         elif command == "/help":
#             return """
# Available commands:
# - /general - Switch to general AI chat mode
# - /house_info - Switch to house information mode
# - /guest_guide - Switch to guest messaging guide mode
# - /use_house:[house name] - Set the current house
# - /help - Show this help message
# - /exit - End the conversation
# """
#         else:
#             return f"Unknown command: {command}. Type /help for available commands."
    
#     def generate_response(self, user_input: str) -> str:
#         """Generate response based on current mode."""
#         user_input = re.sub(r'[^\w\s:/-]', '', user_input.strip())  # Basic sanitization
#         if user_input.startswith("/"):
#             return self.process_command(user_input)
        
#         self.update_chat_history("user", user_input)
#         if self.mode == "general":
#             return self.generate_general_response(user_input)
#         elif self.mode == "house_info":
#             return self.generate_house_info_response(user_input)
#         elif self.mode == "guest_guide":
#             return self.generate_guest_guide_response(user_input)
#         self.mode = "general"
#         return self.generate_general_response(user_input)
    
#     def chat(self) -> None:
#         """Interactive chat loop."""
#         print("House Notes Bot: Hello! Type /help for commands or /exit to end.")
#         while True:
#             user_input = input("\nYou: ").strip()
#             if user_input.lower() == '/exit':
#                 print("House Notes Bot: Goodbye!")
#                 break
            
#             response = self.generate_response(user_input)
#             print(f"\nHouse Notes Bot ({self.mode.replace('_', ' ')} mode): {response}")
#             self.update_chat_history("assistant", response)

# if __name__ == "__main__":
#     # Load environment variables from .env file
#     load_dotenv()
    
#     try:
#         bot = HouseNotesBot()
#         bot.chat()
#     except Exception as e:
#         print(f"Error initializing bot: {str(e)}")