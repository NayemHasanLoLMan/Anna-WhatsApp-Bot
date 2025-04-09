
import os
from pinecone import Pinecone
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Tuple
import time
import re
from dotenv import load_dotenv


class HouseNotesBot:
    def __init__(self):
        # API keys from environment variables
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
        self.current_house = None
        
        # Conversation memory
        self.chat_history = []
        
        # House information cache
        self.house_cache = {}
        self.load_house_names()
        
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text using Gemini with input validation"""
        # Validate input to prevent empty content errors
        if not text or not text.strip():
            print("Warning: Empty input for embedding. Using default query.")
            text = "rental property information"
            
        try:
            embedding_model = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return embedding_model['embedding']
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Return zero vector as fallback
            return [0.0] * 768
    
    def update_chat_history(self, role: str, content: str):
        """Add message to chat history"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.chat_history.append({
            "role": role,
            "content": content,
            "timestamp": timestamp,
            "house": self.current_house
        })
        
        # Keep only recent history (last 10 messages)
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]
    
    def detect_house_name(self, query: str) -> Optional[str]:
        """Enhanced house name detection with fuzzy matching"""
        # First, get all house names if cache is empty
        if not self.house_cache:
            self.load_house_names()
        
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
        
        # If no match and we have a current house, maintain it
        if self.current_house:
            return self.current_house
                
        return None
    
    def query_vector_db(self, query: str, index, filter_dict: Optional[Dict] = None, top_k: int = 12) -> List[Dict[str, Any]]:
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
    
    def fetch_relevant_information(self, query: str) -> Dict[str, Any]:
        """Unified method to fetch information from all sources"""
        result = {
            "house_matches": [],
            "guide_matches": [],
            "detected_house": None,
            "house_switch_msg": "",
            "house_context": "",
            "guide_context": "",
            "sources": []
        }
        
        # Detect house name in query
        detected_house = self.detect_house_name(query)
        result["detected_house"] = detected_house
        
        # Handle house transitions
        if detected_house:
            if self.current_house != detected_house:
                self.current_house = detected_house
                result["house_switch_msg"] = f"Now discussing {self.current_house}."
        
        # Get house information with optional house filter
        if self.current_house:
            # Try with current house first
            filter_dict = {"house_name": {"$eq": self.current_house}}
            result["house_matches"] = self.query_vector_db(query, self.house_index, filter_dict, top_k=15)
            
            # If no results with filter, try without filter
            if not result["house_matches"]:
                print(f"No results for {self.current_house}, trying general search")
                result["house_matches"] = self.query_vector_db(query, self.house_index, None, top_k=15)
        else:
            # No specific house, query without filter
            result["house_matches"] = self.query_vector_db(query, self.house_index, None, top_k=15)
            
            # If we found matches but don't have a current house, check if they're all for the same house
            if result["house_matches"] and not self.current_house:
                house_counts = {}
                for match in result["house_matches"]:
                    house = match['metadata'].get('house_name')
                    if house:
                        house_counts[house] = house_counts.get(house, 0) + 1
                
                # If all or most results are for one house, set it as current
                if house_counts:
                    most_common_house = max(house_counts.items(), key=lambda x: x[1])[0]
                    if house_counts[most_common_house] >= len(result["house_matches"]) * 0.7:  # If 70% or more matches are for one house
                        self.current_house = most_common_house
                        result["house_switch_msg"] = f"Now discussing {self.current_house}."
                        result["detected_house"] = self.current_house
        
        # Get guest guide information
        result["guide_matches"] = self.query_vector_db(query, self.guest_guide_index, None, top_k=15)
        
        # Format contexts from both knowledge sources
        if result["house_matches"]:
            result["house_context"], result["sources"] = self.format_house_matches(result["house_matches"])
        
        if result["guide_matches"]:
            result["guide_context"] = self.format_guest_guide_matches(result["guide_matches"])
        
        return result
    
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
        """Format guest guide matches with emphasis on messaging templates and tone guidelines"""
        # Extract templates, tone guidelines, and general information
        templates = []
        tone_guidelines = []
        general_info = []
        
        seen_content = set()
        
        for match in matches:
            text = match['metadata'].get('text', '').strip()
            if not text or text[:100] in seen_content:
                continue
                
            seen_content.add(text[:100])
            lower_text = text.lower()
            
            # Identify tone and branding guidelines
            if any(marker in lower_text for marker in ["tone", "voice", "brand", "persona", "brindy"]):
                tone_guidelines.append({"text": text, "score": match['score']})
            # Identify message templates
            elif any(marker in lower_text for marker in ["template", "example", "sample", "message"]) or \
                any(marker in text for marker in ["$", "[", "]", "{", "}"]):
                templates.append({"text": text, "score": match['score']})
            else:
                general_info.append({"text": text, "score": match['score']})
        
        # Sort by relevance score
        templates.sort(key=lambda x: x['score'], reverse=True)
        tone_guidelines.sort(key=lambda x: x['score'], reverse=True)
        general_info.sort(key=lambda x: x['score'], reverse=True)
        
        # Build formatted context with clear sections
        formatted_text = ""
        
        if tone_guidelines:
            formatted_text += "## BRINDY'S MESSAGING GUIDELINES:\n\n"
            for item in tone_guidelines[:3]:  # Limit to most relevant
                formatted_text += f"{item['text']}\n\n"
        
        if templates:
            formatted_text += "## MESSAGE TEMPLATES:\n\n"
            for item in templates[:5]:  # Limit to most relevant templates
                formatted_text += f"{item['text']}\n\n"
        
        if general_info:
            formatted_text += "## ADDITIONAL CONTEXT:\n\n"
            for item in general_info[:3]:  # Limit to most relevant
                formatted_text += f"{item['text']}\n\n"
                    
        return formatted_text
    
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
    
    def format_chat_history(self) -> str:
        """Format chat history for context"""
        formatted = []
        for msg in self.chat_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted)
    
    def process_command(self, command: str) -> str:
        """Process special commands"""
        command = command.lower().strip()
        
        if command == "/help":
            return """
                Available commands:
                - /help - Show this help message
                - /reset - Reset conversation and current house
                - /houses - List available houses
                """
        elif command == "/reset":
            self.current_house = None
            self.chat_history = []
            return "Conversation reset. Starting fresh!"
        elif command == "/houses":
            houses = list(set(self.house_cache.values())) if self.house_cache else []
            house_list = ", ".join(houses[:10]) if houses else "No houses found in database"
            if len(houses) > 10:
                house_list += f", and {len(houses) - 10} more"
            return f"Available houses: {house_list}"
        else:
            return f"Unknown command: {command}. Type /help for available commands."
    
    def generate_response(self, user_input: str) -> str:
        """Generate unified response with emphasis on proper guest messaging"""
        # Check if input is a command
        if user_input.startswith("/"):
            return self.process_command(user_input)
        
        # Fetch all relevant information
        info = self.fetch_relevant_information(user_input)
        
        # Determine if we have any useful context
        has_house_info = bool(info["house_context"])
        has_guide_info = bool(info["guide_context"])
        
        # Create a prompt based on available information
        if not has_house_info and not has_guide_info:
            # No relevant information found in either database
            houses = list(set(self.house_cache.values()))
            houses_sample = ", ".join(houses[:5])
            if len(houses) > 5:
                houses_sample += f", and {len(houses) - 5} more"
                
            prompt = f"""You are Brindy, a professional property manager creating messages for guests.
                    
                    Recent conversation:
                    {self.format_chat_history()}

                    User question: {user_input}

                    Create a professional, warm message that could be sent to a guest. Focus on creating a ready-to-send message in Brindy's voice rather than explaining how to create one. Assume a professional but friendly tone suitable for vacation rental communications.
                    
                    If the user is asking about a specific property, mention that you know about these properties: {houses_sample}
                    
                    Response as Brindy:"""
        else:
            # Create comprehensive prompt including all available knowledge
            house_info_section = ""
            if has_house_info:
                house_name = info["detected_house"] if info["detected_house"] else "various properties"
                house_info_section = f"""
                PROPERTY INFORMATION:
                House: {house_name}
                {info["house_context"]}
                """
            
            guide_info_section = ""
            if has_guide_info:
                guide_info_section = f"""
                {info["guide_context"]}
                """
            
            prompt = f"""You are Brindy, a professional property manager who creates messages for guests.
                    
                    Recent conversation:
                    {self.format_chat_history()}

                    IMPORTANT INSTRUCTIONS:
                    1. Create a message AS BRINDY that is ready to be sent to a guest - do not explain what you would write
                    2. Use the proper tone and style specified in the messaging guidelines
                    3. Include relevant property details when available
                    4. Create a complete, professional message that requires no further editing
                    5. Don't include content not relevant to the specific situation
                    6. Don't make up information not present in the provided context
                    
                    Property Context:
                    {house_info_section if house_info_section.strip() else "No specific property information available"}
                    
                    Messaging Guidelines:
                    {guide_info_section if guide_info_section.strip() else "Use professional, warm tone appropriate for vacation rental communications"}

                    User request: {user_input}

                    Write a complete, personalized message as Brindy that could be sent directly to a guest:"""

        try:
            response = self.gemini_model.generate_content(prompt)
            answer = response.text
            
            # Add house switch message if needed
            if info["house_switch_msg"]:
                answer = f"{info['house_switch_msg']}\n\n{answer}"
                    
            # Add sources info if we have house matches (but only if it's not a guest message)
            if info["sources"] and "create" not in user_input.lower() and "write" not in user_input.lower():
                answer += f"\n\n(Information sourced from: {', '.join(info['sources'])})"
                    
            return answer
        except Exception as e:
            return f"I'm sorry, I encountered an error processing your request: {str(e)}"
        


    def create_guest_message(self, query: str, house_info: str = None) -> str:
        """Create a guest message using available information"""
        # Get guest guide information
        guide_matches = self.query_vector_db(query, self.guest_guide_index, None, top_k=15)
        guide_context = self.format_guest_guide_matches(guide_matches)
        
        # Create house context if available
        house_context = ""
        house_name = "the property"
        if house_info and self.current_house:
            house_context = house_info
            house_name = self.current_house
        
        prompt = f"""You are Brindy, a professional property manager creating a message for guests.
        
        MESSAGE REQUEST: {query}
        
        PROPERTY: {house_name}
        {house_context}
        
        MESSAGING GUIDELINES:
        {guide_context}
        
        INSTRUCTIONS:
        1. Create a complete, ready-to-send message AS Brindy (not about Brindy)
        2. Use the tone and style specified in the messaging guidelines
        3. Personalize the message using property details when available
        4. Keep the message concise but comprehensive
        5. Include appropriate greeting and sign-off
        6. Do not explain the message - write it as if sending directly to a guest
        
        Create a professional guest message:"""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error creating guest message: {str(e)}"
        


    def detect_message_type(self, query: str) -> str:
        """Detect what type of response is needed"""
        query_lower = query.lower()
        
        # Patterns that indicate a guest message is requested
        message_patterns = [
            "create a message", "write a message", "draft a", "compose a", 
            "respond to", "reply to", "how should i respond", 
            "message for guest", "template for", "welcome message",
            "check-in message", "check out message"
        ]
        
        # Check if query matches any message pattern
        for pattern in message_patterns:
            if pattern in query_lower:
                return "guest_message"
        
        # Otherwise assume information retrieval
        return "information"
    


    def process_query(self, user_input: str) -> str:
        """Main query processing function with improved message type detection"""
        # Check if input is a command
        if user_input.startswith("/"):
            return self.process_command(user_input)
        
        # Detect if this is a request for a message or information
        message_type = self.detect_message_type(user_input)
        
        # Fetch relevant house information
        info = self.fetch_relevant_information(user_input)
        
        # Handle house switching message
        house_switch_msg = info["house_switch_msg"]
        
        # Generate appropriate response based on message type
        if message_type == "guest_message":
            response = self.create_guest_message(user_input, info["house_context"])
        else:
            response = self.generate_response(user_input)
        
        # Add house switch message if needed
        if house_switch_msg:
            response = f"{house_switch_msg}\n\n{response}"
        
        return response

    
    def chat(self):
        """Interactive chat loop"""
        print("House Notes Bot: Hello! I'm your unified property assistant. I can help with property information and guest messaging.\n"
              "You can ask about specific properties or request guest message templates.\n"
              "Type /help for available commands.")
        
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
            
            print(f"\nHouse Notes Bot: {response}")
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

###################################################### TEST VERSION ######################################################

# import os
# from pinecone import Pinecone
# import google.generativeai as genai
# from typing import List, Dict, Any, Optional, Tuple
# import time
# import re
# from dotenv import load_dotenv


# class HouseNotesBot:
#     def __init__(self):
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
        
#         # Initialize Gemini
#         genai.configure(api_key=self.GOOGLE_API_KEY)
#         self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        
#         # Bot state
#         self.mode = "general"  # Modes: "general", "house_info", "guest_guide"
#         self.current_house = None
        
#         # Conversation memory
#         self.chat_history = []
        
#         # House information cache
#         self.house_cache = {}
#         self.load_house_names()
        
#     def embed_text(self, text: str) -> List[float]:
#         """Generate embeddings for text using Gemini"""
#         try:
#             embedding_model = genai.embed_content(
#                 model="models/embedding-001",
#                 content=text,
#                 task_type="retrieval_document"
#             )
#             return embedding_model['embedding']
#         except Exception as e:
#             print(f"Error generating embeddings: {e}")
#             # Return zero vector as fallback
#             return [0.0] * 768
    
#     def update_chat_history(self, role: str, content: str):
#         """Add message to chat history with house context"""
#         timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
#         self.chat_history.append({
#             "role": role,
#             "content": content,
#             "timestamp": timestamp,
#             "mode": self.mode,
#             "house": self.current_house if self.mode == "house_info" else None
#         })
        
#         # Keep only recent history (last 10 messages)
#         if len(self.chat_history) > 10:
#             self.chat_history = self.chat_history[-10:]
    
#     def detect_house_name(self, query: str) -> Optional[str]:
#         """Enhanced house name detection with fuzzy matching"""
#         # First, get all house names if cache is empty
#         if not self.house_cache:
#             self.load_house_names()
        
#         # Check if query contains any house name
#         query_lower = query.lower().strip()
#         query_no_space = query_lower.replace(" ", "")
        
#         # Try exact matches first
#         for house_key, house_name in self.house_cache.items():
#             if house_key == query_lower or house_key in query_lower:
#                 return house_name
        
#         # Try fuzzy matching for similar names
#         for house_key, house_name in self.house_cache.items():
#             # Check if query without spaces matches house key without spaces
#             if house_key.replace(" ", "") == query_no_space:
#                 return house_name
            
#             # Check for significant substring matches (at least 70% of the house name)
#             if len(house_key) > 4 and (house_key in query_lower or query_lower in house_key):
#                 return house_name
        
#         # If no match and we have a current house, maintain it
#         if self.current_house:
#             return self.current_house
                
#         return None
    
#     def query_vector_db(self, query: str, index, filter_dict: Optional[Dict] = None, top_k: int = 12) -> List[Dict[str, Any]]:
#         """Query specified Pinecone index with filtering options"""
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
    
#     def query_all_databases(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
#         """Query both house info and guest guide databases for general mode"""
#         # Query house information first
#         house_matches = self.query_vector_db(query, self.house_index, None, top_k=8)
        
#         # Query guest guide information
#         guide_matches = self.query_vector_db(query, self.guest_guide_index, None, top_k=8)
        
#         return {
#             "house_matches": house_matches,
#             "guide_matches": guide_matches
#         }
    
#     def query_house_info(self, query: str, house_name: Optional[str] = None) -> List[Dict[str, Any]]:
#         """Query house information with optional house name filter"""
#         # Try with house filter first if provided
#         if house_name:
#             filter_dict = {"house_name": {"$eq": house_name}}
#             results = self.query_vector_db(query, self.house_index, filter_dict, top_k=15)
            
#             # If no results with filter, try without filter
#             if not results:
#                 # Log that we're trying without filter
#                 print(f"No results for {house_name}, trying general search")
#                 results = self.query_vector_db(query, self.house_index, None, top_k=15)
                
#             return results
#         else:
#             # No house specified, try general search
#             return self.query_vector_db(query, self.house_index, None, top_k=15)
    
#     def query_guest_guide(self, query: str) -> List[Dict[str, Any]]:
#         """Query guest messaging guide with increased result count"""
#         return self.query_vector_db(query, self.guest_guide_index, top_k=15)
    
#     def format_chat_history(self) -> str:
#         """Format chat history for context"""
#         formatted = []
#         current_mode_history = [msg for msg in self.chat_history if msg["mode"] == self.mode]
        
#         for msg in current_mode_history:
#             role = "User" if msg["role"] == "user" else "Assistant"
#             formatted.append(f"{role}: {msg['content']}")
        
#         return "\n".join(formatted)
    
#     def format_house_matches(self, matches: List[Dict]) -> Tuple[str, List[str]]:
#         """Format house info matches into context string and source references"""
#         context_chunks = []
#         sources = []
        
#         for match in matches:
#             document_name = match['metadata'].get('file_name', 'Unknown document')
#             text = match['metadata'].get('text', '')
#             house_name = match['metadata'].get('house_name', 'Unknown house')
            
#             if text:
#                 context_chunks.append(f"[Source: {document_name}, House: {house_name}]\n{text}")
#                 if document_name not in sources:
#                     sources.append(f"{document_name} (House: {house_name})")
        
#         return "\n\n".join(context_chunks), sources
    
#     def format_guest_guide_matches(self, matches: List[Dict]) -> str:
#         """Format guest guide matches into comprehensive context string focused on templates and detailed information"""
#         # Extract and categorize text chunks
#         templates = []
#         detailed_info = []
        
#         # Track seen content to avoid duplicates
#         seen_content = set()
        
#         for match in matches:
#             text = match['metadata'].get('text', '').strip()
#             if not text or text[:100] in seen_content:
#                 continue
                
#             # Add to seen content
#             seen_content.add(text[:100])
            
#             # Categorize the content - focus on identifying templates
#             lower_text = text.lower()
            
#             # Check if it's a template (typically has placeholders or specific formatting)
#             is_template = (
#                 any(marker in lower_text for marker in ["template", "example message", "sample response"]) or
#                 any(marker in text for marker in ["$", "[", "]", "{", "}"]) or
#                 text.count("\n") <= 3  # Short paragraphs are likely templates
#             )
            
#             if is_template:
#                 templates.append({"text": text, "score": match['score']})
#             else:
#                 detailed_info.append({"text": text, "score": match['score']})
        
#         # Sort each category by relevance score
#         templates.sort(key=lambda x: x['score'], reverse=True)
#         detailed_info.sort(key=lambda x: x['score'], reverse=True)
        
#         # Format with organized sections - prioritizing templates
#         formatted_text = ""
        
#         if templates:
#             formatted_text += "## MESSAGE TEMPLATES:\n\n"
#             for item in templates:
#                 formatted_text += f"{item['text']}\n\n"
        
#         if detailed_info:
#             formatted_text += "## DETAILED INFORMATION:\n\n"
#             for item in detailed_info:
#                 formatted_text += f"{item['text']}\n\n"
                
#         return formatted_text
    
#     def load_house_names(self):
#         """Pre-load house names at startup"""
#         try:
#             print("Loading house names...")
#             # Query index with a dummy vector to get metadata
#             dummy_vector = [0.0] * 768
#             results = self.house_index.query(vector=dummy_vector, top_k=100, include_metadata=True)
            
#             # Extract unique house names
#             houses_found = 0
#             for match in results.get('matches', []):
#                 house_name = match['metadata'].get('house_name')
#                 if house_name and house_name != "Unknown" and house_name.lower() not in self.house_cache:
#                     self.house_cache[house_name.lower()] = house_name
#                     houses_found += 1
                    
#                     # Also store without spaces and special characters
#                     no_space = house_name.lower().replace(" ", "")
#                     self.house_cache[no_space] = house_name
                    
#                     simple_name = re.sub(r'[^a-zA-Z0-9]', '', house_name.lower())
#                     if simple_name != house_name.lower():
#                         self.house_cache[simple_name] = house_name
            
#             print(f"Loaded {houses_found} unique houses into cache")
#             return houses_found > 0
#         except Exception as e:
#             print(f"Error loading house names: {e}")
#             return False
    
#     def generate_general_response(self, query: str) -> str:
#         """Generate enhanced general response that incorporates all knowledge sources"""
#         # Query both knowledge bases
#         all_results = self.query_all_databases(query)
#         house_matches = all_results["house_matches"]
#         guide_matches = all_results["guide_matches"]
        
#         # Format contexts from both knowledge sources
#         house_context = ""
#         sources = []
        
#         if house_matches:
#             house_context, sources = self.format_house_matches(house_matches)
        
#         guide_context = ""
#         if guide_matches:
#             guide_context = self.format_guest_guide_matches(guide_matches)
        
#         # Determine if we have any useful context
#         has_context = bool(house_context or guide_context)
        
#         if not has_context:
#             # No relevant information found in either database
#             prompt = f"""You are a helpful AI assistant that specializes in rental property management and guest messaging.
            
#                     Recent conversation:
#                     {self.format_chat_history()}
    
#                     User question: {query}
    
#                     Respond to the user's question based on your general knowledge about rental properties and guest messaging best practices.
#                     Be helpful, conversational, and friendly. If you need specific property information to answer accurately, 
#                     suggest that the user switch to house information mode using /house_info command.
                    
#                     Response:"""
#         else:
#             # Create comprehensive prompt including all available knowledge
#             prompt = f"""You are a helpful AI assistant that specializes in rental property management and guest messaging.
                    
#                     Recent conversation:
#                     {self.format_chat_history()}
    
#                     Use the following information to answer the latest question. Be conversational and friendly.
#                     Include specific details when available but don't make up information.
    
#                     Property Information:
#                     {house_context}
    
#                     Guest Messaging Guidelines:
#                     {guide_context}
    
#                     User question: {query}
    
#                     Response (providing helpful, accurate information and being transparent about knowledge limitations):"""
    
#         try:
#             response = self.gemini_model.generate_content(prompt)
#             answer = response.text
            
#             # Add sources info if we have house matches
#             if sources:
#                 answer += f"\n\n(Information sourced from: {', '.join(sources)})"
                    
#             return answer
#         except Exception as e:
#             return f"I'm sorry, I encountered an error processing your question: {str(e)}"
    
#     def generate_house_info_response(self, query: str) -> str:
#         """Generate enhanced response about house information with improved house detection and comprehensive information"""
#         # Detect house name in query
#         detected_house = self.detect_house_name(query)
        
#         # Handle house transitions
#         house_switch_msg = ""
#         if detected_house:
#             if self.current_house != detected_house:
#                 self.current_house = detected_house
#                 house_switch_msg = f"Now discussing {self.current_house}."
        
#         # Get relevant documents with expanded top_k for more comprehensive information
#         matches = []
#         if self.current_house:
#             # Try with current house first
#             matches = self.query_house_info(query, self.current_house)
        
#         # If no matches or no current house, try general search
#         if not matches:
#             matches = self.query_house_info(query)
            
#             # If we found matches but don't have a current house, check if they're all for the same house
#             if matches and not self.current_house:
#                 house_counts = {}
#                 for match in matches:
#                     house = match['metadata'].get('house_name')
#                     if house:
#                         house_counts[house] = house_counts.get(house, 0) + 1
                
#                 # If all or most results are for one house, set it as current
#                 if house_counts:
#                     most_common_house = max(house_counts.items(), key=lambda x: x[1])[0]
#                     if house_counts[most_common_house] >= len(matches) * 0.7:  # If 70% or more matches are for one house
#                         self.current_house = most_common_house
#                         house_switch_msg = f"Now discussing {self.current_house}."
        
#         # Format context from matches
#         context, sources = self.format_house_matches(matches)
        
#         if not context:
#             houses = list(set(self.house_cache.values()))
#             houses_sample = ", ".join(houses[:5])
#             if len(houses) > 5:
#                 houses_sample += f", and {len(houses) - 5} more"
                
#             if self.current_house:
#                 return f"I don't have specific information about that for {self.current_house}. Please try asking something else about the property."
#             else:
#                 return f"I don't have information about that. Available houses include: {houses_sample}"
        
#         # Create enhanced prompt for Gemini with more specific instructions
#         house_context = f"Current house being discussed: {self.current_house}" if self.current_house else "No specific house selected"
#         prompt = f"""You are an expert on rental properties, specializing in providing detailed information about specific houses.
                
#                 {house_context}

#                 Recent conversation:
#                 {self.format_chat_history()}

#                 IMPORTANT INSTRUCTIONS:
#                 1. Provide COMPREHENSIVE details about the property based on the information below
#                 2. Include specific amenities, features, locations, and unique characteristics
#                 3. Mention the house name ({self.current_house if self.current_house else "appropriate house"}) explicitly in your response
#                 4. Organize your response in a structured way with clear sections if appropriate
#                 5. Be conversational and engaging, as if you're a property manager enthusiastically describing the property
#                 6. Do NOT make up information that isn't in the provided context
#                 7. Include ALL relevant details from the information provided

#                 Property Information:
#                 {context}

#                 User question: {query}

#                 Comprehensive, detailed response about the property:"""

#         try:
#             response = self.gemini_model.generate_content(prompt)
#             answer = response.text
            
#             # Add house switch message and sources if needed
#             if house_switch_msg:
#                 answer = f"{house_switch_msg}\n\n{answer}"
                    
#             # Add sources info if we have matches
#             if sources:
#                 answer += f"\n\n(Information sourced from: {', '.join(sources)})"
                    
#             return answer
#         except Exception as e:
#             return f"I'm sorry, I encountered an error retrieving house information: {str(e)}"
    
#     def generate_guest_guide_response(self, query: str) -> str:
#         """Generate enhanced response based on guest messaging guide with improved template implementation"""
#         # Get expanded set of relevant documents from guest guide
#         matches = self.query_guest_guide(query)
        
#         if not matches:
#             return "I don't have specific guest messaging guidelines for that query. Please try asking something else or switch to another mode."
        
#         # Format context from matches with improved categorization
#         context = self.format_guest_guide_matches(matches)
        
#         # Create improved prompt for Gemini with specific instructions for guest guide responses
#         prompt = f"""You are a professional host messaging specialist. Your task is to create a COMPLETE, READY-TO-SEND message for hosts to use with their guests.

#             Recent conversation:
#             {self.format_chat_history()}

#             IMPORTANT INSTRUCTIONS:
#             1. Create a COMPLETE, POLISHED message that follows the instructional tone and style from the guide
#             2. The message should be ready to copy and send to guests WITHOUT any additional editing needed
#             3. Use appropriate templates from the guide, adapting them to the specific situation
#             4. Include appropriate greetings, closings, and all necessary details
#             5. Maintain a warm, professional, and hospitable tone throughout
#             6. Format the message with proper spacing, paragraphs, and structure for easy readability
#             7. After providing the complete message, add a brief section titled "Message Notes" with any relevant context about why you formatted the message this way
#             8. The primary goal is to create a message that hosts can immediately send to guests without modification

#             Guest messaging guide content:
#             {context}

#             User request: {query}

#             Complete, ready-to-send guest message:"""

#         try:
#             response = self.gemini_model.generate_content(prompt)
#             return response.text
#         except Exception as e:
#             return f"I'm sorry, I encountered an error creating the guest message: {str(e)}"
    
#     def process_command(self, command: str) -> str:
#         """Process special commands"""
#         command = command.lower().strip()
        
#         if command == "/general":
#             self.mode = "general"
#             return "Switched to general mode. I'll respond to your questions using all available knowledge about properties and guest messaging."
            
#         elif command == "/house_info":
#             self.mode = "house_info"
#             houses = list(set(self.house_cache.values())) if self.house_cache else []
#             house_list = ", ".join(houses[:5]) if houses else "No houses found in database"
#             if len(houses) > 5:
#                 house_list += f", and {len(houses) - 5} more"
#             return f"Switched to house information mode. I'll provide comprehensive details about properties.\nSome available houses: {house_list}"
            
#         elif command == "/guest_guide":
#             self.mode = "guest_guide"
#             return "Switched to guest messaging mode. I'll create professional, ready-to-send messages based on the guest messaging guidelines."
            
#         elif command == "/help":
#             return """
#                 Available commands:
#                 - /general - Switch to general mode with access to all knowledge
#                 - /house_info - Switch to house information mode for detailed property information
#                 - /guest_guide - Switch to guest messaging mode for professional guest communication
#                 - /help - Show this help message
#                 """
#         else:
#             return f"Unknown command: {command}. Type /help for available commands."
    
#     def generate_response(self, user_input: str) -> str:
#         """Generate response based on current mode"""
#         # Check if input is a command
#         if user_input.startswith("/"):
#             return self.process_command(user_input)
        
#         # Generate response based on current mode
#         if self.mode == "general":
#             return self.generate_general_response(user_input)
#         elif self.mode == "house_info":
#             return self.generate_house_info_response(user_input)
#         elif self.mode == "guest_guide":
#             return self.generate_guest_guide_response(user_input)
#         else:
#             self.mode = "general"  # Reset to general mode if unknown
#             return self.generate_general_response(user_input)
    
#     def chat(self):
#         """Interactive chat loop"""
#         print("House Notes Bot: Hello! I can operate in different modes:\n"
#               "- General mode (default) - Access to all knowledge\n"
#               "- House information mode (/house_info) - Detailed property information\n" 
#               "- Guest messaging mode (/guest_guide) - Create professional guest messages\n"
#               "Type /help for all commands.")
        
#         while True:
#             user_input = input("\nYou: ").strip()
            
#             if user_input.lower() == '/exit':
#                 print("House Notes Bot: Goodbye!")
#                 break
                
#             # Update chat history
#             self.update_chat_history("user", user_input)
            
#             # Generate and display response
#             start_time = time.time()
#             response = self.generate_response(user_input)
#             end_time = time.time()
            
#             print(f"\nHouse Notes Bot ({self.mode.replace('_', ' ')} mode): {response}")
#             print(f"[Response time: {end_time - start_time:.2f}s]")
            
#             # Update chat history with bot response
#             self.update_chat_history("assistant", response)


# if __name__ == "__main__":
#     # Load environment variables from .env file
#     load_dotenv()
    
#     try:
#         bot = HouseNotesBot()
#         bot.chat()
#     except Exception as e:
#         print(f"Error initializing bot: {str(e)}")














################################################################ STABLE VERSION ##################################



# import os
# from pinecone import Pinecone
# import google.generativeai as genai
# from typing import List, Dict, Any, Optional, Tuple
# import time
# import re
# from dotenv import load_dotenv


# class HouseNotesBot:
#     def __init__(self):
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
        
#         # Initialize Gemini
#         genai.configure(api_key=self.GOOGLE_API_KEY)
#         self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        
#         # Bot state
#         self.current_house = None
        
#         # Conversation memory
#         self.chat_history = []
        
#         # House information cache
#         self.house_cache = {}
#         self.load_house_names()
        
#     def embed_text(self, text: str) -> List[float]:
#         """Generate embeddings for text using Gemini with input validation"""
#         # Validate input to prevent empty content errors
#         if not text or not text.strip():
#             print("Warning: Empty input for embedding. Using default query.")
#             text = "rental property information"
            
#         try:
#             embedding_model = genai.embed_content(
#                 model="models/embedding-001",
#                 content=text,
#                 task_type="retrieval_document"
#             )
#             return embedding_model['embedding']
#         except Exception as e:
#             print(f"Error generating embeddings: {e}")
#             # Return zero vector as fallback
#             return [0.0] * 768
    
#     def update_chat_history(self, role: str, content: str):
#         """Add message to chat history"""
#         timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
#         self.chat_history.append({
#             "role": role,
#             "content": content,
#             "timestamp": timestamp,
#             "house": self.current_house
#         })
        
#         # Keep only recent history (last 10 messages)
#         if len(self.chat_history) > 10:
#             self.chat_history = self.chat_history[-10:]
    
#     def detect_house_name(self, query: str) -> Optional[str]:
#         """Enhanced house name detection with fuzzy matching"""
#         # First, get all house names if cache is empty
#         if not self.house_cache:
#             self.load_house_names()
        
#         # Check if query contains any house name
#         query_lower = query.lower().strip()
#         query_no_space = query_lower.replace(" ", "")
        
#         # Try exact matches first
#         for house_key, house_name in self.house_cache.items():
#             if house_key == query_lower or house_key in query_lower:
#                 return house_name
        
#         # Try fuzzy matching for similar names
#         for house_key, house_name in self.house_cache.items():
#             # Check if query without spaces matches house key without spaces
#             if house_key.replace(" ", "") == query_no_space:
#                 return house_name
            
#             # Check for significant substring matches (at least 70% of the house name)
#             if len(house_key) > 4 and (house_key in query_lower or query_lower in house_key):
#                 return house_name
        
#         # If no match and we have a current house, maintain it
#         if self.current_house:
#             return self.current_house
                
#         return None
    
#     def query_vector_db(self, query: str, index, filter_dict: Optional[Dict] = None, top_k: int = 12) -> List[Dict[str, Any]]:
#         """Query specified Pinecone index with filtering options"""
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
    
#     def fetch_relevant_information(self, query: str) -> Dict[str, Any]:
#         """Unified method to fetch information from all sources"""
#         result = {
#             "house_matches": [],
#             "guide_matches": [],
#             "detected_house": None,
#             "house_switch_msg": "",
#             "house_context": "",
#             "guide_context": "",
#             "sources": []
#         }
        
#         # Detect house name in query
#         detected_house = self.detect_house_name(query)
#         result["detected_house"] = detected_house
        
#         # Handle house transitions
#         if detected_house:
#             if self.current_house != detected_house:
#                 self.current_house = detected_house
#                 result["house_switch_msg"] = f"Now discussing {self.current_house}."
        
#         # Get house information with optional house filter
#         if self.current_house:
#             # Try with current house first
#             filter_dict = {"house_name": {"$eq": self.current_house}}
#             result["house_matches"] = self.query_vector_db(query, self.house_index, filter_dict, top_k=15)
            
#             # If no results with filter, try without filter
#             if not result["house_matches"]:
#                 print(f"No results for {self.current_house}, trying general search")
#                 result["house_matches"] = self.query_vector_db(query, self.house_index, None, top_k=15)
#         else:
#             # No specific house, query without filter
#             result["house_matches"] = self.query_vector_db(query, self.house_index, None, top_k=15)
            
#             # If we found matches but don't have a current house, check if they're all for the same house
#             if result["house_matches"] and not self.current_house:
#                 house_counts = {}
#                 for match in result["house_matches"]:
#                     house = match['metadata'].get('house_name')
#                     if house:
#                         house_counts[house] = house_counts.get(house, 0) + 1
                
#                 # If all or most results are for one house, set it as current
#                 if house_counts:
#                     most_common_house = max(house_counts.items(), key=lambda x: x[1])[0]
#                     if house_counts[most_common_house] >= len(result["house_matches"]) * 0.7:  # If 70% or more matches are for one house
#                         self.current_house = most_common_house
#                         result["house_switch_msg"] = f"Now discussing {self.current_house}."
#                         result["detected_house"] = self.current_house
        
#         # Get guest guide information
#         result["guide_matches"] = self.query_vector_db(query, self.guest_guide_index, None, top_k=15)
        
#         # Format contexts from both knowledge sources
#         if result["house_matches"]:
#             result["house_context"], result["sources"] = self.format_house_matches(result["house_matches"])
        
#         if result["guide_matches"]:
#             result["guide_context"] = self.format_guest_guide_matches(result["guide_matches"])
        
#         return result
    
#     def format_house_matches(self, matches: List[Dict]) -> Tuple[str, List[str]]:
#         """Format house info matches into context string and source references"""
#         context_chunks = []
#         sources = []
        
#         for match in matches:
#             document_name = match['metadata'].get('file_name', 'Unknown document')
#             text = match['metadata'].get('text', '')
#             house_name = match['metadata'].get('house_name', 'Unknown house')
            
#             if text:
#                 context_chunks.append(f"[Source: {document_name}, House: {house_name}]\n{text}")
#                 if document_name not in sources:
#                     sources.append(f"{document_name} (House: {house_name})")
        
#         return "\n\n".join(context_chunks), sources
    
#     def format_guest_guide_matches(self, matches: List[Dict]) -> str:
#         """Format guest guide matches into comprehensive context string focused on templates and detailed information"""
#         # Extract and categorize text chunks
#         templates = []
#         detailed_info = []
        
#         # Track seen content to avoid duplicates
#         seen_content = set()
        
#         for match in matches:
#             text = match['metadata'].get('text', '').strip()
#             if not text or text[:100] in seen_content:
#                 continue
                
#             # Add to seen content
#             seen_content.add(text[:100])
            
#             # Categorize the content - focus on identifying templates
#             lower_text = text.lower()
            
#             # Check if it's a template (typically has placeholders or specific formatting)
#             is_template = (
#                 any(marker in lower_text for marker in ["template", "example message", "sample response"]) or
#                 any(marker in text for marker in ["$", "[", "]", "{", "}"]) or
#                 text.count("\n") <= 3  # Short paragraphs are likely templates
#             )
            
#             if is_template:
#                 templates.append({"text": text, "score": match['score']})
#             else:
#                 detailed_info.append({"text": text, "score": match['score']})
        
#         # Sort each category by relevance score
#         templates.sort(key=lambda x: x['score'], reverse=True)
#         detailed_info.sort(key=lambda x: x['score'], reverse=True)
        
#         # Format with organized sections - prioritizing templates
#         formatted_text = ""
        
#         if templates:
#             formatted_text += "## MESSAGE TEMPLATES:\n\n"
#             for item in templates:
#                 formatted_text += f"{item['text']}\n\n"
        
#         if detailed_info:
#             formatted_text += "## DETAILED INFORMATION:\n\n"
#             for item in detailed_info:
#                 formatted_text += f"{item['text']}\n\n"
                
#         return formatted_text
    
#     def load_house_names(self):
#         """Pre-load house names at startup"""
#         try:
#             print("Loading house names...")
#             # Query index with a dummy vector to get metadata
#             dummy_vector = [0.0] * 768
#             results = self.house_index.query(vector=dummy_vector, top_k=100, include_metadata=True)
            
#             # Extract unique house names
#             houses_found = 0
#             for match in results.get('matches', []):
#                 house_name = match['metadata'].get('house_name')
#                 if house_name and house_name != "Unknown" and house_name.lower() not in self.house_cache:
#                     self.house_cache[house_name.lower()] = house_name
#                     houses_found += 1
                    
#                     # Also store without spaces and special characters
#                     no_space = house_name.lower().replace(" ", "")
#                     self.house_cache[no_space] = house_name
                    
#                     simple_name = re.sub(r'[^a-zA-Z0-9]', '', house_name.lower())
#                     if simple_name != house_name.lower():
#                         self.house_cache[simple_name] = house_name
            
#             print(f"Loaded {houses_found} unique houses into cache")
#             return houses_found > 0
#         except Exception as e:
#             print(f"Error loading house names: {e}")
#             return False
    
#     def format_chat_history(self) -> str:
#         """Format chat history for context"""
#         formatted = []
#         for msg in self.chat_history:
#             role = "User" if msg["role"] == "user" else "Assistant"
#             formatted.append(f"{role}: {msg['content']}")
        
#         return "\n".join(formatted)
    
#     def process_command(self, command: str) -> str:
#         """Process special commands"""
#         command = command.lower().strip()
        
#         if command == "/help":
#             return """
#                 Available commands:
#                 - /help - Show this help message
#                 - /reset - Reset conversation and current house
#                 - /houses - List available houses
#                 """
#         elif command == "/reset":
#             self.current_house = None
#             self.chat_history = []
#             return "Conversation reset. Starting fresh!"
#         elif command == "/houses":
#             houses = list(set(self.house_cache.values())) if self.house_cache else []
#             house_list = ", ".join(houses[:10]) if houses else "No houses found in database"
#             if len(houses) > 10:
#                 house_list += f", and {len(houses) - 10} more"
#             return f"Available houses: {house_list}"
#         else:
#             return f"Unknown command: {command}. Type /help for available commands."
    
#     def generate_response(self, user_input: str) -> str:
#         """Generate unified response considering all information sources"""
#         # Check if input is a command
#         if user_input.startswith("/"):
#             return self.process_command(user_input)
        
#         # Fetch all relevant information
#         info = self.fetch_relevant_information(user_input)
        
#         # Determine if we have any useful context
#         has_house_info = bool(info["house_context"])
#         has_guide_info = bool(info["guide_context"])
        
#         # Create a prompt based on available information
#         if not has_house_info and not has_guide_info:
#             # No relevant information found in either database
#             houses = list(set(self.house_cache.values()))
#             houses_sample = ", ".join(houses[:5])
#             if len(houses) > 5:
#                 houses_sample += f", and {len(houses) - 5} more"
                
#             prompt = f"""You are a helpful AI assistant that specializes in rental property management and guest messaging.
                    
#                     Recent conversation:
#                     {self.format_chat_history()}
    
#                     User question: {user_input}
    
#                     Respond to the user's question based on your general knowledge about rental properties and guest messaging best practices.
#                     Be helpful, conversational, and friendly. 
                    
#                     If the user is asking about a specific property, mention that you can provide information about these properties: {houses_sample}
                    
#                     Response:"""
#         else:
#             # Create comprehensive prompt including all available knowledge
#             house_info_section = ""
#             if has_house_info:
#                 house_name = info["detected_house"] if info["detected_house"] else "various properties"
#                 house_info_section = f"""
#                 PROPERTY INFORMATION:
#                 House: {house_name}
#                 {info["house_context"]}
#                 """
            
#             guide_info_section = ""
#             if has_guide_info:
#                 guide_info_section = f"""
#                 GUEST MESSAGING GUIDELINES:
#                 {info["guide_context"]}
#                 """
            
#             prompt = f"""You are a comprehensive property management and guest messaging assistant.
                    
#                     Recent conversation:
#                     {self.format_chat_history()}
    
#                     IMPORTANT INSTRUCTIONS:
#                     1. You provide BOTH property information AND professional guest messaging assistance in one comprehensive response
#                     2. When replying about properties, be descriptive and include all relevant details
#                     3. When creating guest messages, make them complete and ready to send
#                     4. If the user is asking about a property, prioritize property details
#                     5. If the user is asking for a message template, prioritize creating a professional message
#                     6. Don't make up information not present in the provided context
                    
#                     RESPONSE GUIDELINES:
#                     1. Keep responses focused and concise
#                     2. Prioritize information based on user's specific question
#                     3. For property queries:
#                     - Focus on key features and relevant details
#                     - Highlight unique selling points
#                     - Include location and amenity information if asked
                    
#                     4. For guest messages:
#                     - Create personalized, ready-to-send messages
#                     - Adapt templates to specific context
#                     - Include property details when available
#                     - Keep tone warm and professional
                    
#                     5. When both property and messaging are relevant:
#                     - Start with brief property highlights
#                     - Follow with a tailored guest message
#                     - Include only essential additional context

#                 Available Context:
#                 {house_info_section if house_info_section.strip() else "No specific property information available"}
                
#                 Guest Guide Context:
#                 {guide_info_section if guide_info_section.strip() else "Using general guest communication best practices"}

#                 User question: {user_input}

#                 Provide a concise, relevant response focusing on what's most important for this specific query:"""
    
#         try:
#             response = self.gemini_model.generate_content(prompt)
#             answer = response.text
            
#             # Add house switch message if needed
#             if info["house_switch_msg"]:
#                 answer = f"{info['house_switch_msg']}\n\n{answer}"
                    
#             # Add sources info if we have house matches
#             if info["sources"]:
#                 answer += f"\n\n(Information sourced from: {', '.join(info['sources'])})"
                    
#             return answer
#         except Exception as e:
#             return f"I'm sorry, I encountered an error processing your question: {str(e)}"
    
#     def chat(self):
#         """Interactive chat loop"""
#         print("House Notes Bot: Hello! I'm your unified property assistant. I can help with property information and guest messaging.\n"
#               "You can ask about specific properties or request guest message templates.\n"
#               "Type /help for available commands.")
        
#         while True:
#             user_input = input("\nYou: ").strip()
            
#             if user_input.lower() == '/exit':
#                 print("House Notes Bot: Goodbye!")
#                 break
                
#             # Update chat history
#             self.update_chat_history("user", user_input)
            
#             # Generate and display response
#             start_time = time.time()
#             response = self.generate_response(user_input)
#             end_time = time.time()
            
#             print(f"\nHouse Notes Bot: {response}")
#             print(f"[Response time: {end_time - start_time:.2f}s]")
            
#             # Update chat history with bot response
#             self.update_chat_history("assistant", response)


# if __name__ == "__main__":
#     # Load environment variables from .env file
#     load_dotenv()
    
#     try:
#         bot = HouseNotesBot()
#         bot.chat()
#     except Exception as e:
#         print(f"Error initializing bot: {str(e)}")