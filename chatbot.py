import os
from pinecone import Pinecone
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Tuple
import time
import re
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

class AnnaAssistant:
    def __init__(self):
        # API keys from environment variables
        self.PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
        self.GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
        
        # Index names
        self.HOUSE_INDEX_NAME = "house-information-embeddings"
        self.GUEST_GUIDE_INDEX_NAME = "information-massageing-guide-embeddings"
        
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
        
        # Property-related keywords
        self.property_keywords = [
            "check-in", "check-out", "amenities", "access", "wi-fi", "password", "rules", 
            "pets", "smoking", "parking", "location", "neighborhood", "transportation", 
            "fees", "deposit", "damage", "cleaning", "maintenance", "alarms", "security", 
            "keys", "codes", "complaints", "issues", "problems", "questions", "information", 
            "details"
        ]
        
        # Reminder functionality
        self.reminders = []
        self.reminder_trigger_keywords = ["/note", "/remind", "/reminder"]
        
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text using Gemini with input validation"""
        # Validate input to prevent empty content errors
        if not text or not text.strip():
            print("Warning: Empty input for embedding. Using default query.")
            text = "rental property information"
            
        try:
            embedding_model = genai.embed_content(
                model="models/text-embedding-004",
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
        """Enhanced house name detection with improved context preservation"""
        if not self.house_cache:
            self.load_house_names()
        
        query_lower = query.lower().strip()
        query_no_space = query_lower.replace(" ", "")
        
        # Step 1: Check for explicit house name matches
        for house_key, house_name in self.house_cache.items():
            if house_key == query_lower or house_key in query_lower:
                return house_name
            if house_key.replace(" ", "") == query_no_space:
                return house_name
            if len(house_key) > 4 and (house_key in query_lower or query_lower in house_key):
                return house_name
        
        # Step 2: Handle short or ambiguous inputs by checking conversation history
        if len(query_lower) <= 3 or query_lower.isdigit():  # For inputs like "23"
            # Check recent messages for house context
            for msg in reversed(self.chat_history[-3:]):  # Look at the last 3 messages
                if msg["house"]:
                    # If the short input matches part of the last house (e.g., "23" in "Heatherbrae 23")
                    if query_lower in msg["house"].lower():
                        return msg["house"]
            # If no match in history, maintain current house
            if self.current_house:
                return self.current_house
        
        # Step 3: Maintain current house if no new house is detected
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
        """Unified method to fetch information with better house context handling"""
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
        
        # Check if query contains property keywords
        query_lower = query.lower()
        contains_property_keyword = any(keyword in query_lower for keyword in self.property_keywords)
        
        # Fetch house matches only if:
        # - A house name is detected in the query, or
        # - The query contains property keywords and a current_house is set
        if detected_house:
            self.current_house = detected_house
            
            filter_dict = {"house_name": {"$eq": self.current_house}}
            result["house_matches"] = self.query_vector_db(query, self.house_index, filter_dict, top_k=15)
        elif contains_property_keyword and self.current_house:
            filter_dict = {"house_name": {"$eq": self.current_house}}
            result["house_matches"] = self.query_vector_db(query, self.house_index, filter_dict, top_k=15)
        
        # Always fetch guide matches
        result["guide_matches"] = self.query_vector_db(query, self.guest_guide_index, None, top_k=15)
        
        # Format contexts
        if result["house_matches"]:
            result["house_context"], result["sources"] = self.format_house_matches(result["house_matches"])
        
        if result["guide_matches"]:
            result["guide_context"] = self.format_guest_guide_matches(result["guide_matches"], query)
        
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
    
    def format_guest_guide_matches(self, matches: List[Dict], query: str = "") -> str:
        """Format guest guide matches with query-specific prioritization"""
        # Extract templates, tone guidelines, and general information
        templates = []
        tone_guidelines = []
        general_info = []
        
        seen_content = set()
        query_lower = query.lower()
        
        for match in matches:
            text = match['metadata'].get('text', '').strip()
            if not text or text[:100] in seen_content:
                continue
            seen_content.add(text[:100])
            lower_text = text.lower()
            
            # Categorize based on content markers
            if any(marker in lower_text for marker in ["template", "example", "sample", "message"]) or \
               any(marker in text for marker in ["$", "[", "]", "{", "}"]):
                templates.append({"text": text, "score": match['score']})
            elif any(marker in lower_text for marker in ["tone", "voice", "brand", "persona"]):
                tone_guidelines.append({"text": text, "score": match['score']})
            else:
                general_info.append({"text": text, "score": match['score']})
        
        # Determine query intent to prioritize sections
        is_drafting_message = any(keyword in query_lower for keyword in ["draft", "message", "respond", "reply"])
        is_policy_guidance = any(keyword in query_lower for keyword in ["book", "policy", "contact", "directly"])
        
        # Sort by relevance score
        templates.sort(key=lambda x: x['score'], reverse=True)
        tone_guidelines.sort(key=lambda x: x['score'], reverse=True)
        general_info.sort(key=lambda x: x['score'], reverse=True)
        
        # Build formatted context based on query intent
        formatted_text = ""
        
        if is_drafting_message:
            # Prioritize templates and tone for drafting messages
            if templates:
                formatted_text += "## MESSAGE TEMPLATES & EXAMPLES (Use these for drafting):\n\n"
                for item in templates[:3]:  # Limit to top 3 most relevant
                    formatted_text += f"{item['text']}\n\n"
            if tone_guidelines:
                formatted_text += "## MESSAGING TONE & BRAND GUIDELINES (Follow this tone):\n\n"
                for item in tone_guidelines[:2]:  # Limit to top 2
                    formatted_text += f"{item['text']}\n\n"
            if general_info:
                formatted_text += "## GENERAL MESSAGING GUIDELINES (Additional context):\n\n"
                for item in general_info[:2]:
                    formatted_text += f"{item['text']}\n\n"
        
        elif is_policy_guidance:
            # Prioritize tone and general guidelines for policy-related queries
            if tone_guidelines:
                formatted_text += "## MESSAGING TONE & BRAND GUIDELINES (Follow this tone):\n\n"
                for item in tone_guidelines[:2]:
                    formatted_text += f"{item['text']}\n\n"
            if general_info:
                formatted_text += "## GENERAL MESSAGING GUIDELINES (Use for policy guidance):\n\n"
                for item in general_info[:3]:
                    formatted_text += f"{item['text']}\n\n"
            if templates:
                formatted_text += "## MESSAGE TEMPLATES & EXAMPLES (Optional for reference):\n\n"
                for item in templates[:2]:
                    formatted_text += f"{item['text']}\n\n"
        
        else:
            # Default ordering for other queries
            if tone_guidelines:
                formatted_text += "## MESSAGING TONE & BRAND GUIDELINES:\n\n"
                for item in tone_guidelines[:2]:
                    formatted_text += f"{item['text']}\n\n"
            if templates:
                formatted_text += "## MESSAGE TEMPLATES & EXAMPLES:\n\n"
                for item in templates[:3]:
                    formatted_text += f"{item['text']}\n\n"
            if general_info:
                formatted_text += "## GENERAL MESSAGING GUIDELINES:\n\n"
                for item in general_info[:2]:
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
            role = "Employee" if msg["role"] == "user" else "Anna"
            formatted.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted)
    
    def generate_response(self, user_input: str) -> str:
        """Unified response generation with strict adherence to messaging guide"""
        # Check if this is a reminder request
        if self.is_reminder_request(user_input):
            return self.process_reminder(user_input)
            
        info = self.fetch_relevant_information(user_input)
        
        house_context = info["house_context"]
        house_name = info["detected_house"] if info["detected_house"] else "the property"
        guide_context = info["guide_context"]
        formatted_history = self.format_chat_history()
        
        prompt = f"""You are Anna, a helpful AI assistant for property management employees. Your role is to provide accurate, tailored responses based on the provided property details, messaging guidelines, and your general knowledge. Use the context to craft clear, specific, and explanatory answers that directly address the employee's query without quoting or reproducing raw information from the context. Synthesize the information to provide natural, colleague-like responses that include relevant context and explanations.

            EMPLOYEE QUERY: {user_input}

            Recent conversation:
            {formatted_history}

            INSTRUCTIONS:
            1. Analyze the query to identify the employee's needs (e.g., drafting a message, seeking property details, or requesting communication guidance).
            2. For property-related queries, extract specific details from PROPERTY INFORMATION and present them clearly and concisely. Interpret terms like 'structure,' 'layout,' or 'rooms' as referring to the physical attributes of the house (e.g., floor plan, number of bedrooms, bathrooms) unless otherwise specified.
            3. If no relevant information is found in PROPERTY INFORMATION for a property-related query, explicitly state that no information is available for the requested details (e.g., 'No information is available about the layout of {house_name}.').
            4. For communication tasks (e.g., drafting messages or providing tone advice), strictly adhere to the tone, structure, and content specified in the MESSAGING GUIDELINES. Explain why the chosen tone or structure is appropriate, referencing the guidelines implicitly.
            5. When drafting messages, adapt MESSAGE TEMPLATES & EXAMPLES to the specific context (e.g., property name, issue), and provide a brief explanation of how the message aligns with the guidelines.
            6. If the query is ambiguous (e.g., unclear intent or multiple possible interpretations), ask clarifying questions, using PROPERTY INFORMATION or recent conversation to inform the clarification (e.g., 'Are you asking about the layout of {house_name} or something else?').
            7. Incorporate relevant background or situational context to make responses more comprehensive, but keep them focused and professional.
            8. Do not invent information beyond what is provided in the context or your knowledge.
            9. Maintain a helpful, colleague-like tone, avoiding overly formal or robotic language.
            10. Ensure responses are concise, precise, and directly address the query without unnecessary details.
            11. If no specific property information is available for a house-related query, prioritize general guidelines from the MESSAGING GUIDELINES section to provide a helpful response based on standard practices.
            12. For house related quries, don't give all the information at once. Instead, provide a summary of the most relevant details and offer to provide more information if needed.
            13. Answare precisely and avoid unnecessary repetition or filler content.
            14. for general queries, provide a summary of the most relevant details and offer to provide more information if needed.
            15. if no house name is given within the query use general massage templates and examples to provide a helpful response based on standard practices.



            PROPERTY INFORMATION ({house_name}):
            {house_context if house_context.strip() else "No specific property information available for this query."}

            MESSAGING GUIDELINES:
            {guide_context if guide_context.strip() else "No specific messaging guidelines available for this query."}

            Provide a response that addresses the employee's needs, weaving in relevant context and explanations while strictly adhering to the messaging guidelines for communication-related tasks:"""

        try:
            response = self.gemini_model.generate_content(prompt)
            answer = response.text
            
            # Add house switch message if needed
            if info["house_switch_msg"]:
                answer = f"{info['house_switch_msg']}\n\n{answer}"
            
            return answer
        except Exception as e:
            return f"I'm sorry, I encountered an error processing your request: {str(e)}"
    
    # Updated and improved reminder functionality methods
    def is_reminder_request(self, user_input: str) -> bool:
        """Check if the query is a reminder request"""
        user_input_lower = user_input.lower().strip()
        
        # Check for trigger keywords at the beginning of the message
        for keyword in self.reminder_trigger_keywords:
            if user_input_lower.startswith(keyword):
                return True
                
        # Check for reminder phrases
        reminder_phrases = [
            "remind me", 
            "set a reminder", 
            "create a reminder",
            "set a note",
            "make a note",
            "add a reminder"
        ]
        
        for phrase in reminder_phrases:
            if phrase in user_input_lower:
                return True
                
        return False
        
    def parse_reminder_time(self, text: str) -> tuple:
        """
        Extract time and date information from natural language text with improved pattern recognition
        Returns a tuple of (date_str, time_str, natural_date)
        """
        text_lower = text.lower()
        print(f"Parsing reminder time from: '{text_lower}'")  # Debug logging
        
        # Extract specified time (enhanced patterns)
        time_patterns = [
            r'(\d{1,2})[\.:](\d{2})(?:\s*([ap]m))?',      # matches "5:00", "5.00", "5:00 am"
            r'(\d{1,2})\s*([ap]m)',                       # matches "5 am", "5am", "5 pm"
            r'at\s+(\d{1,2})(?:[\.:](\d{2}))?(?:\s*([ap]m))?'  # matches "at 5", "at 5:00", "at 5.00"
        ]
        
        time_str = None
        for pattern in time_patterns:
            matches = re.search(pattern, text_lower)
            if matches:
                groups = matches.groups()
                
                # Handle hour
                hour = int(groups[0])
                
                # Handle minutes if present, otherwise default to 0
                minutes = 0
                if len(groups) > 1 and groups[1] and groups[1].isdigit():
                    minutes = int(groups[1])
                
                # Check for PM/AM in the matched groups and surrounding context
                is_pm = False
                is_am = False
                
                # Check the actual match for AM/PM
                for g in groups:
                    if g and isinstance(g, str):
                        if "pm" in g.lower():
                            is_pm = True
                            break
                        elif "am" in g.lower():
                            is_am = True
                            break
                
                # If not found in groups, check surrounding context
                if not is_pm and not is_am:
                    # Get the entire sentence or segment containing the time
                    sentence_pattern = r'[^.!?]*' + re.escape(matches.group(0)) + r'[^.!?]*'
                    sentence_match = re.search(sentence_pattern, text_lower)
                    
                    if sentence_match:
                        context = sentence_match.group(0)
                        if "pm" in context or "p.m" in context or "evening" in context:
                            is_pm = True
                        elif "am" in context or "a.m" in context or "morning" in context:
                            is_am = True
                
                # Apply AM/PM conversion
                if is_pm and hour < 12:
                    hour += 12
                elif is_am and hour == 12:
                    hour = 0  # Convert 12 AM to 00 hours
                
                time_str = f"{hour:02d}:{minutes:02d}"
                print(f"Extracted time: {hour}:{minutes:02d} (is_pm: {is_pm}, is_am: {is_am})")
                break
        
        # Date extraction with enhanced pattern recognition
        now = datetime.now()
        date_str = "today"  # Default to today
        natural_date = now.strftime("%Y-%m-%d")  # Default formatted date
        
        # Dictionary of month names to month numbers
        month_names = {
            "january": 1, "jan": 1,
            "february": 2, "feb": 2,
            "march": 3, "mar": 3,
            "april": 4, "apr": 4,
            "may": 5,
            "june": 6, "jun": 6,
            "july": 7, "jul": 7,
            "august": 8, "aug": 8,
            "september": 9, "sep": 9, "sept": 9,
            "october": 10, "oct": 10,
            "november": 11, "nov": 11,
            "december": 12, "dec": 12
        }
        
        # First, check for ordinal dates like "23rd of May" - prioritize this pattern
        ordinal_date_pattern = r'(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(' + '|'.join(list(month_names.keys())) + r')'
        ordinal_match = re.search(ordinal_date_pattern, text_lower)
        
        if ordinal_match:
            day_num, month_name = ordinal_match.groups()
            month_num = month_names[month_name]
            day_num = int(day_num)
            
            # Set year (current or next if the date has passed)
            target_year = now.year
            if (month_num < now.month) or (month_num == now.month and day_num < now.day):
                target_year += 1
                
            specific_date = datetime(target_year, month_num, day_num)
            date_str = f"on {month_name} {day_num}"
            natural_date = specific_date.strftime("%Y-%m-%d")
            
            # If no time was found, set a default based on context
            if not time_str:
                if "morning" in text_lower:
                    time_str = "09:00"
                elif "afternoon" in text_lower:
                    time_str = "14:00"
                elif "evening" in text_lower or "tonight" in text_lower:
                    time_str = "19:00"
                else:
                    time_str = "09:00"  # Default to 9 AM
                    
            print(f"Parsed date from ordinal: {natural_date}, display: {date_str}")
            
        # Handle "today", "tomorrow", "next week" patterns
        elif "today" in text_lower or "toay" in text_lower:  # Handle common misspelling
            date_str = "today"
            natural_date = now.strftime("%Y-%m-%d")
            print(f"Parsed date as today: {natural_date}")
        
        elif "tomorrow" in text_lower or "tommorow" in text_lower:  # Handle common misspelling
            date_str = "tomorrow"
            natural_date = (now + timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"Parsed date as tomorrow: {natural_date}")
        
        # Dictionary of day names to weekday numbers (0=Monday, 6=Sunday)
        days_of_week = {
            "monday": 0, "mon": 0,
            "tuesday": 1, "tue": 1, "tues": 1,
            "wednesday": 2, "wed": 2,
            "thursday": 3, "thu": 3, "thurs": 3,
            "friday": 4, "fri": 4,
            "saturday": 5, "sat": 5,
            "sunday": 6, "sun": 6
        }
        
        # Handle day names (e.g., "on Monday", "this Friday")
        for day_name, day_num in days_of_week.items():
            if day_name in text_lower:
                # Calculate days until the next occurrence of this weekday
                current_weekday = now.weekday()
                days_until = (day_num - current_weekday) % 7
                
                # If we're referring to a day that's already past this week
                if days_until == 0 and "next" in text_lower:
                    days_until = 7  # Next week's same day
                elif days_until == 0:
                    days_until = 7  # Default to next week if the day is today
                
                target_date = now + timedelta(days=days_until)
                date_str = f"on {day_name}"
                natural_date = target_date.strftime("%Y-%m-%d")
                print(f"Parsed date from day name: {natural_date}, display: {date_str}")
                break
        
        # Handle "next week" patterns
        if "next week" in text_lower:
            # Default to next Monday if day not specified
            next_monday = now + timedelta(days=(7 - now.weekday()))
            date_str = "next week"
            natural_date = next_monday.strftime("%Y-%m-%d")
            print(f"Parsed date as next week: {natural_date}")
        
        # Handle specific date formats like "on May 15"
        specific_date_pattern = r'(?:on\s+)?(' + '|'.join(month_names.keys()) + r')\s+(\d{1,2})(?:st|nd|rd|th)?'
        date_match = re.search(specific_date_pattern, text_lower)
        
        if date_match:
            month_name, day = date_match.groups()
            month_num = month_names[month_name]
            day_num = int(day)
            
            # Set year (current or next if the date has passed)
            target_year = now.year
            if (month_num < now.month) or (month_num == now.month and day_num < now.day):
                target_year += 1
                
            specific_date = datetime(target_year, month_num, day_num)
            date_str = f"on {month_name} {day}"
            natural_date = specific_date.strftime("%Y-%m-%d")
            print(f"Parsed date from month-day: {natural_date}, display: {date_str}")
        
        # If no time was found, set a default based on context
        if not time_str:
            if "morning" in text_lower:
                time_str = "09:00"
            elif "afternoon" in text_lower:
                time_str = "14:00"
            elif "evening" in text_lower or "tonight" in text_lower:
                time_str = "19:00"
            else:
                time_str = "09:00"  # Default to 9 AM
            print(f"Using default time: {time_str}")
        
        print(f"Final parsed time: {time_str}, date: {natural_date}, display: {date_str}")
        return date_str, time_str, natural_date
    
    def extract_reminder_content(self, text: str) -> str:
        """Extract the actual reminder content/task from the input"""
        # Remove trigger keywords
        content = text
        for keyword in self.reminder_trigger_keywords:
            if content.lower().startswith(keyword.lower()):
                content = content[len(keyword):].strip()
                break
        
        # Remove time references
        time_phrases = [
            r'at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?',
            r'\d{1,2}(?::\d{2})?\s*(?:am|pm)',
            r'this\s+(?:morning|afternoon|evening)',
            r'tomorrow(?:\s+morning|\s+afternoon|\s+evening)?',
            r'next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
            r'on\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
            r'next\s+week',
            r'remind\s+me',
            r'set\s+a\s+reminder',
            r'create\s+a\s+reminder',
            r'make\s+a\s+note'
        ]
        
        for phrase in time_phrases:
            content = re.sub(phrase, '', content, flags=re.IGNORECASE)
        
        # Clean up content
        content = re.sub(r'\s+', ' ', content).strip()
        
        # If the content starts with common action words, keep them
        # Otherwise, if needed, add a starter phrase to make it a complete note
        if not content:
            return "General reminder"
        return content
    
    def generate_ai_reminder_message(self, raw_content: str, date_str: str, time_str: str) -> str:
        """Use Gemini AI to generate a professional reminder message"""
        try:
            # Extract time info for context
            hour, minute = map(int, time_str.split(':'))
            am_pm = "AM" if hour < 12 else "PM"
            display_hour = hour % 12
            if display_hour == 0:
                display_hour = 12
            formatted_time = f"{display_hour}:{minute:02d} {am_pm}"
            
            # Create prompt for Gemini
            prompt = f"""
            Please format the following reminder content into a professional, well-structured reminder message:
            
            Original reminder: "{raw_content}"
            Time: {formatted_time}
            Date: {date_str}
            
            Requirements:
            1. Create a bold title that captures the main purpose (use ** for bold)
            2. Write a concise, professional body with relevant details
            3. Include an appropriate closing phrase
            4. Keep the message friendly and professional
            5. Format as a short, clear paragraph (2-3 sentences maximum)
            6. Start with the title on its own line, followed by the message body
            7. Don't explicitly mention this is an "AI-generated" reminder
            8. Don't include timestamps, references to "original reminder", or meta-information
            
            Example format:
            **Maintenance Appointment**
            Your maintenance appointment is scheduled as planned. Please have all necessary documents ready. Have a productive day!
            """
            
            # Generate the formatted message
            response = self.gemini_model.generate_content(prompt)
            formatted_message = response.text.strip()
            
            # Fallback in case of AI failure
            if not formatted_message or len(formatted_message) < 10:
                return f"**Reminder**\n{raw_content}"
                
            return formatted_message
            
        except Exception as e:
            print(f"Error generating AI reminder message: {e}")
            # Fallback in case of error
            return f"**Reminder**\n{raw_content}"
    
    def process_reminder(self, user_input: str) -> str:
        """Process reminder request and generate appropriate response with AI-enhanced formatting"""
        # Parse reminder details
        date_str, time_str, natural_date = self.parse_reminder_time(user_input)
        
        # Extract reminder content (the basic content)
        raw_content = self.extract_reminder_content(user_input)
        
        # Generate AI-formatted message
        formatted_message = self.generate_ai_reminder_message(raw_content, date_str, time_str)
        
        # Format time for the reminder
        hour, minute = map(int, time_str.split(':'))
        reminder_time = f"{hour:02d}:{minute:02d}"
        
        # Create the reminder JSON
        reminder = {
            "send_to": "self",
            "time": reminder_time,
            "date": natural_date,
            "message": formatted_message
        }
        
        # Store this reminder for future reference
        self.reminders.append(reminder)
        
        # Format time for response
        am_pm = "AM" if hour < 12 else "PM"
        display_hour = hour % 12
        if display_hour == 0:
            display_hour = 12
            
        formatted_time = f"{display_hour}:{minute:02d} {am_pm}"
        
        # Parse the date for display
        reminder_date = datetime.strptime(reminder["date"], "%Y-%m-%d").date()
        today = datetime.now().date()
        
        if reminder_date == today:
            date_display = "today"
        elif reminder_date == today + timedelta(days=1):
            date_display = "tomorrow"
        else:
            # Format as weekday, Month Day
            date_display = reminder_date.strftime("%A, %B %d")
        
        # Generate response message
        response = f"I've set a reminder for {date_display} at {formatted_time}.\n" 
        
        
        # Add JSON for debugging or API integration
        reminder_json = json.dumps(reminder, indent=2)
        response += f"\n\nReminder details:\n```json\n{reminder_json}\n```"
          
        return response
    
    def process_query(self, user_input: str) -> str:
        """Main query processing function"""
        # Generate response
        response = self.generate_response(user_input)
        
        # Update chat history
        self.update_chat_history("user", user_input)
        self.update_chat_history("assistant", response)
        
        return response
    
    def chat(self):
        """Interactive chat loop"""
        print("Anna: Hello! I'm Anna, your property assistant. I am here to help you find information about properties and messaging guidelines.")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == '/exit':
                print("Anna: Goodbye!")
                break
                
            # Generate and display response
            start_time = time.time()
            response = self.process_query(user_input)
            end_time = time.time()
            
            print(f"\nAnna: {response}")
            print(f"[Response time: {end_time - start_time:.2f}s]")



def process_backend_query(conversation_history: list, user_input: str) -> dict:
    """
    Process a query from the backend API, with support for conversation history.
    
    Args:
        conversation_history: List of message dictionaries with 'role' and 'content' keys
                              where role is either 'user' or 'assistant'
        user_input: The current user message
    
    Returns:
        Dictionary containing:
            - 'response': The assistant's response text (without JSON details for reminders)
            - 'is_reminder': Boolean indicating if this is a reminder
            - 'reminder_data': JSON reminder data (if applicable, else None)
    """
    try:
        # Initialize the assistant
        assistant = AnnaAssistant()
        
        # Load conversation history
        for message in conversation_history:
            if 'role' in message and 'content' in message:
                assistant.update_chat_history(message['role'], message['content'])
        
        # Check if this is a reminder request
        is_reminder = assistant.is_reminder_request(user_input)
        reminder_data = None
        
        if is_reminder:
            # Parse reminder details
            date_str, time_str, natural_date = assistant.parse_reminder_time(user_input)
            
            # Extract reminder content
            raw_content = assistant.extract_reminder_content(user_input)
            
            # Generate AI-formatted message
            formatted_message = assistant.generate_ai_reminder_message(raw_content, date_str, time_str)
            
            # Format time for the reminder
            hour, minute = map(int, time_str.split(':'))
            reminder_time = f"{hour:02d}:{minute:02d}"
            
            # Create the reminder data
            reminder_data = {
                "send_to": "self",
                "time": reminder_time,
                "date": natural_date,
                "message": formatted_message
            }
            
            # Format time for response
            am_pm = "AM" if hour < 12 else "PM"
            display_hour = hour % 12
            if display_hour == 0:
                display_hour = 12
                
            formatted_time = f"{display_hour}:{minute:02d} {am_pm}"
            
            # Parse the date for display
            reminder_date = datetime.strptime(reminder_data["date"], "%Y-%m-%d").date()
            today = datetime.now().date()
            
            if reminder_date == today:
                date_display = "today"
            elif reminder_date == today + timedelta(days=1):
                date_display = "tomorrow"
            else:
                # Format as weekday, Month Day
                date_display = reminder_date.strftime("%A, %B %d")
            
            # Generate user-facing response message (without JSON details)
            response = f"I've set a reminder for {date_display} at {formatted_time}."
            
        else:
            # Process regular query
            response = assistant.generate_response(user_input)
        
        # Save this interaction to history
        assistant.update_chat_history("user", user_input)
        assistant.update_chat_history("assistant", response)
        
        return {
            "response": response,
            "is_reminder": is_reminder,
            "reminder_data": reminder_data
        }
    
    except Exception as e:
        error_message = f"Error processing backend query: {str(e)}"
        return {
            "response": error_message,
            "is_reminder": False,
            "reminder_data": None
        }

# if __name__ == "__main__":
#     # Load environment variables from .env file
    
#     try:
#         assistant = AnnaAssistant()
#         assistant.chat()
#     except Exception as e:
#         print(f"Error initializing assistant: {str(e)}")





if __name__ == "__main__":
    # Example usage
    sample_history = [
        {"role": "user", "content": "What's the check-in time for Heatherbrae 23?"},
        {"role": "assistant", "content": "The check-in time for Heatherbrae 23 is 3:00 PM. Please remember to provide the access code to guests."}
    ]
    sample_input = "set a reminder for toay at 10.20 for the check-in time for Heatherbrae 23"
    
    # Process the query using the backend integration function
    response = process_backend_query(sample_history, sample_input)
    

    print(f"Message: {response['response']}")
    print(json.dumps(response['reminder_data'], indent=2))
    