
import os
from pinecone import Pinecone
import google.generativeai as genai
from typing import List, Dict, Any
from datetime import datetime, timedelta
import re
import json
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
import openai


load_dotenv()

try:
    has_rich = True
    console = Console()
except ImportError:
    has_rich = False

class AlanaAssistant:
    def __init__(self):
        # API keys from environment variables
        self.PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
        # self.GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
        self.OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.PINECONE_API_KEY)
        self.house_index = self.pc.Index("brindy-house-test-knowladgebase")
        self.guest_guide_index = self.pc.Index("brindy-guest-test-knowladgebase")
        
        # Initialize Gemini
        # genai.configure(api_key=self.GOOGLE_API_KEY)
        # self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        
        openai.api_key = self.OPENAI_API_KEY
        self.embedding_model = "text-embedding-ada-002"  # Updated to newer model
        self.chat_model = "gpt-4-turbo"  # Updated to newer model


        # Reminders storage
        self.reminders = []

        # Add house name patterns for better extraction
        self.house_name_patterns = [
            r'\b(?:for|at|in|about)\s+(?:the\s+)?([a-zA-Z0-9\s]+?)(?:\s+house|\s+property|\?|$)',
            r'\b([a-zA-Z0-9\s]+?)(?:\s+house|\s+property)',
            r'^(?:what|how|when|where|tell me about)\s+.*?(?:for|at|in|about)\s+(?:the\s+)?([a-zA-Z0-9\s]+?)(?:\?|$)',
        ] 
        
        # Reminder trigger keywords
        self.reminder_trigger_keywords = ["/note", "/remind", "/reminder"]



    def extract_house_name(self, query: str) -> str:
        """Extract house name from user query."""
        if not query:
            return None
            
        query_clean = query.strip().lower()
        
        # Direct lookup in known houses first (most reliable)
        known_houses = [
            '81st way desert rose', 'arcadia', 'camelback casita 63rd pi', 'casa coconino',
            'elmerville hummingbird crossing', 'granada house', 'heatherbrae 1', 'heatherbrae 23',
            'kenwood', 'kysar cabin', 'mesa coastal', 'navajo flats', 
            'newport beach 1, 2, 3', 'paradise', 'siesta pacifica'
        ]
        
        # Check for exact matches first
        for house in known_houses:
            if house in query_clean:
                return house.title()
        
        # Try pattern matching as fallback - but be more strict
        for pattern in self.house_name_patterns:
            matches = re.search(pattern, query_clean, re.IGNORECASE)
            if matches:
                house_name = matches.group(1).strip()
                # Clean up common words
                house_name = re.sub(r'\b(the|a|an|at|for|to|from|in|on)\b', '', house_name).strip()
                
                # Only return if it's a reasonable length and matches known houses
                if len(house_name) > 2:
                    # Check if extracted name matches any known house (partial match)
                    for known_house in known_houses:
                        # Use more lenient matching for partial names
                        if (house_name in known_house or 
                            any(word in known_house for word in house_name.split() if len(word) > 2)):
                            return known_house.title()
        
        # If no valid house name found, return None
        return None

    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text using OpenAI."""
        if not text or not text.strip():
            text = "rental property information"
        try:
            response = openai.Embedding.create(
                model=self.embedding_model,
                input=text
            )
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return [0.0] * 1536  # embedding size for ada-002
    
    def query_vector_db(self, query: str, index, house_name: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
            """Query Pinecone index with the given query, optionally filtered by house name."""
            query_vector = self.embed_text(query)
            
            try:
                # Build the query parameters
                query_params = {
                    "vector": query_vector,
                    "top_k": top_k * 2,  # Get more results for better filtering
                    "include_metadata": True
                }
                
                # Query without filter first (since exact matching might be tricky)
                results = index.query(**query_params)
                matches = results.get('matches', [])
                
                # If house name is provided, filter results manually for better control
                if house_name and matches:
                    filtered_matches = []
                    house_name_lower = house_name.lower()
                    
                    for match in matches:
                        metadata = match.get('metadata', {})
                        stored_house_name = metadata.get('house_name', '').lower()
                        file_name = metadata.get('file_name', '').lower()
                        
                        # Multiple matching strategies
                        match_found = False
                        
                        # Strategy 1: Exact match
                        if stored_house_name == house_name_lower:
                            match_found = True
                        
                        # Strategy 2: Check if query house name is contained in stored name
                        elif house_name_lower in stored_house_name:
                            match_found = True
                        
                        # Strategy 3: Check if stored name is contained in query house name
                        elif stored_house_name in house_name_lower:
                            match_found = True
                        
                        # Strategy 4: Word-by-word matching for compound names
                        elif any(word in stored_house_name for word in house_name_lower.split() if len(word) > 2):
                            match_found = True
                        
                        # Strategy 5: Check filename pattern (fallback)
                        elif house_name_lower.replace(' ', '') in file_name.replace(' ', '').replace('_', ''):
                            match_found = True
                        
                        if match_found:
                            filtered_matches.append(match)
                    
                    return filtered_matches[:top_k]
                
                return matches[:top_k]
                
            except Exception as e:
                print(f"Error querying index: {e}")
                return []
    
    def format_matches(self, matches: List[Dict], query: str, house_name: str = None) -> str:
        """Format matches into a context string with house identification."""
        if not matches:
            if house_name:
                return f"No relevant information found for {house_name}."
            return "No relevant information found."
        
        context = []
        for match in matches:
            text = match['metadata'].get('text', '')
            source = match['metadata'].get('file_name', 'Unknown document')
            house_from_metadata = match['metadata'].get('house_name', '')
            
            if text:
                # Use the house name from metadata for clarity
                house_identifier = house_from_metadata if house_from_metadata else "Unknown Property"
                context.append(f"[Property: {house_identifier}]\n{text}")
        
        return "\n\n".join(context)
    
    def format_conversation_history(self, conversation_history: List[Dict[str, str]]) -> str:
        """Format conversation history for inclusion in the prompt."""
        if not conversation_history:
            return ""
        
        formatted_history = []
        for message in conversation_history:
            role = message.get('role', '').capitalize()
            content = message.get('content', '')
            if role and content:
                formatted_history.append(f"{role}: {content}")
        
        return "\n".join(formatted_history)
    
    def is_reminder_request(self, user_input: str) -> bool:
        """Check if the query is a reminder request."""
        user_input_lower = user_input.lower().strip()
        for keyword in self.reminder_trigger_keywords:
            if user_input_lower.startswith(keyword):
                return True
        reminder_phrases = [
            "remind me", "set a reminder", "create a reminder",
            "set a note", "make a note", "add a reminder"
        ]
        return any(phrase in user_input_lower for phrase in reminder_phrases)
    
    def parse_reminder_time(self, text: str) -> tuple:
        """Extract time and date from natural language text."""
        text_lower = text.lower()
        
        # Time patterns
        time_patterns = [
            r'(\d{1,2})[\.:](\d{2})(?:\s*([ap]m))?',
            r'(\d{1,2})\s*([ap]m)',
            r'at\s+(\d{1,2})(?:[\.:](\d{2}))?(?:\s*([ap]m))?'
        ]
        time_str = None
        for pattern in time_patterns:
            matches = re.search(pattern, text_lower)
            if matches:
                groups = matches.groups()
                hour = int(groups[0])
                minutes = 0 if len(groups) < 2 or not groups[1] or not groups[1].isdigit() else int(groups[1])
                is_pm = False
                is_am = False
                for g in groups:
                    if g and isinstance(g, str):
                        if "pm" in g.lower():
                            is_pm = True
                            break
                        elif "am" in g.lower():
                            is_am = True
                            break
                if not is_pm and not is_am:
                    context = re.search(r'[^.!?]*' + re.escape(matches.group(0)) + r'[^.!?]*', text_lower)
                    if context:
                        context = context.group(0)
                        if "pm" in context or "evening" in context:
                            is_pm = True
                        elif "am" in context or "morning" in context:
                            is_am = True
                if is_pm and hour < 12:
                    hour += 12
                elif is_am and hour == 12:
                    hour = 0
                time_str = f"{hour:02d}:{minutes:02d}"
                break
        
        # Date extraction
        now = datetime.now()
        date_str = "today"
        natural_date = now.strftime("%Y-%m-%d")
        
        month_names = {
            "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
            "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
            "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
            "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12
        }
        
        # Handle common misspellings like "toay" for "today"
        if "toay" in text_lower or "today" in text_lower:
            date_str = "today"
            natural_date = now.strftime("%Y-%m-%d")
        elif "tomorrow" in text_lower or "tommorow" in text_lower:
            date_str = "tomorrow"
            natural_date = (now + timedelta(days=1)).strftime("%Y-%m-%d")
        
        ordinal_date_pattern = r'(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(' + '|'.join(month_names.keys()) + r')'
        ordinal_match = re.search(ordinal_date_pattern, text_lower)
        if ordinal_match:
            day_num, month_name = ordinal_match.groups()
            month_num = month_names[month_name]
            day_num = int(day_num)
            target_year = now.year
            if (month_num < now.month) or (month_num == now.month and day_num < now.day):
                target_year += 1
            specific_date = datetime(target_year, month_num, day_num)
            date_str = f"on {month_name} {day_num}"
            natural_date = specific_date.strftime("%Y-%m-%d")
        
        days_of_week = {
            "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1,
            "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3, "thurs": 3,
            "friday": 4, "fri": 4, "saturday": 5, "sat": 5, "sunday": 6, "sun": 6
        }
        for day_name, day_num in days_of_week.items():
            if day_name in text_lower:
                current_weekday = now.weekday()
                days_until = (day_num - current_weekday) % 7
                if days_until == 0 and "next" in text_lower:
                    days_until = 7
                elif days_until == 0:
                    days_until = 7
                target_date = now + timedelta(days=days_until)
                date_str = f"on {day_name}"
                natural_date = target_date.strftime("%Y-%m-%d")
                break
        
        if "next week" in text_lower:
            next_monday = now + timedelta(days=(7 - now.weekday()))
            date_str = "next week"
            natural_date = next_monday.strftime("%Y-%m-%d")
        
        specific_date_pattern = r'(?:on\s+)?(' + '|'.join(month_names.keys()) + r')\s+(\d{1,2})(?:st|nd|rd|th)?'
        date_match = re.search(specific_date_pattern, text_lower)
        if date_match:
            month_name, day = date_match.groups()
            month_num = month_names[month_name]
            day_num = int(day)
            target_year = now.year
            if (month_num < now.month) or (month_num == now.month and day_num < now.day):
                target_year += 1
            specific_date = datetime(target_year, month_num, day_num)
            date_str = f"on {month_name} {day}"
            natural_date = specific_date.strftime("%Y-%m-%d")
        
        if not time_str:
            if "morning" in text_lower:
                time_str = "09:00"
            elif "afternoon" in text_lower:
                time_str = "14:00"
            elif "evening" in text_lower or "tonight" in text_lower:
                time_str = "19:00"
            else:
                time_str = "09:00"
        
        return date_str, time_str, natural_date
    
    def extract_reminder_content(self, text: str, conversation_history=None) -> str:
        """Extract the actual reminder content from the input, optionally using conversation history for context."""
        content = text
        
        # Remove command triggers first
        for keyword in self.reminder_trigger_keywords:
            if content.lower().startswith(keyword.lower()):
                content = content[len(keyword):].strip()
                break
        
        # Remove time-related phrases - expanded list of patterns
        time_phrases = [
            r'(?:for|on)\s+(?:today|toay|tomorrow|tommorow)',
            r'at\s+\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)?',
            r'\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)',
            r'this\s+(?:morning|afternoon|evening|night)',
            r'(?:tomorrow|tommorow)(?:\s+morning|\s+afternoon|\s+evening|\s+night)?',
            r'next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week)',
            r'on\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
            r'(?:remind|reminder|note)\s+(?:me|for|to)',
            r'set\s+a\s+reminder',
            r'create\s+a\s+reminder',
            r'make\s+a\s+note'
        ]
        
        # Apply each pattern sequentially
        for phrase in time_phrases:
            content = re.sub(phrase, '', content, flags=re.IGNORECASE)
        
        # Remove phrases like "for me to" or "for me"
        content = re.sub(r'for\s+me\s+to', '', content, flags=re.IGNORECASE)
        content = re.sub(r'for\s+me', '', content, flags=re.IGNORECASE)
        
        # Clean up whitespace, including multiple spaces and leading/trailing spaces
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Remove any leading conjunctions or prepositions that might be leftover
        content = re.sub(r'^(?:to|and|that|for|about)\s+', '', content, flags=re.IGNORECASE)
        
        # If content is empty or too short, try to extract context from conversation history
        if not content or len(content) < 3:
            if conversation_history and len(conversation_history) > 0:
                # Look for context in the last message from the assistant
                for message in reversed(conversation_history):
                    if message.get('role') == 'assistant':
                        assistant_msg = message.get('content', '')
                        # Extract the main topic from the assistant's message
                        key_phrases = re.findall(r'(?:for|about|regarding|on)\s+(.*?)(?:\.|\!|\?|$)', assistant_msg)
                        if key_phrases:
                            content = key_phrases[0].strip()
                            break
        
        return content if content else "General reminder"
    
    def generate_ai_reminder_message(self, raw_content: str, date_str: str, time_str: str) -> str:
        """Generate a professional reminder message using Gemini."""
        try:
            hour, minute = map(int, time_str.split(':'))
            am_pm = "AM" if hour < 12 else "PM"
            display_hour = hour % 12 or 12
            formatted_time = f"{display_hour}:{minute:02d} {am_pm}"
            
            prompt = f"""
                Create a brief notification message for the following reminder. 
                This message WILL BE DELIVERED at the scheduled time, so write it accordingly:
        
                
                Task/Reminder: "{raw_content}"
                
                Requirements:
                1. Create a bold, attention-grabbing title that clearly identifies what the reminder is for (use * for bold in WhatsApp Markdown)
                2. Write the body in future tense (e.g., "Your meeting will begin in 30 minutes")
                3. Include specific details from the original reminder content
                4. Keep the message friendly, concise, and actionable (2-3 sentences maximum)
                5. Format as a short notification compatible with WhatsApp
                6. Include a gentle call-to-action or helpful tip related to the reminder when appropriate
                7. Write as if this message will appear as a notification at the scheduled time
                8. Start with the title on its own line, followed by the message body
                9. Use WhatsApp Markdown formatting as specified below

                WHATSAPP MARKDOWN FORMATTING:
                - Use `*text*` for bold text to highlight key points, headings, or important terms (e.g., `*Check-in Time*`)
                - Use `_text_` for italic text to emphasize specific details or add a friendly tone (e.g., `_Happy to help!_`)
                - Use `~text~` for strikethrough if indicating something is no longer relevant (e.g., `~Old code: 1234~`)
                - Use triple backticks (```) for code blocks when sharing technical details like access codes or JSON data (e.g., ```Code: 1234```)
                - Use plain text with `-` for bullet points to list information clearly (e.g., `- Item 1\n- Item 2`)
                - Use newlines (`\n`) to separate sections or paragraphs for readability
                - Avoid unsupported Markdown like tables, blockquotes, or links (e.g., `[text](url)`). For links, provide the URL as plain text

                Example format for a meeting reminder:
                *Team Meeting Reminder*\n
                Your scheduled team meeting will begin in 30 minutes. Please have your quarterly report ready to share with the group.

                Example format for a task reminder:
                *Property Inspection Due*\n
                Time to complete the scheduled inspection for 123 Main Street. Remember to bring the updated checklist and take photos as required.
                """
            
            response = openai.ChatCompletion.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant creating reminder notifications."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7,
            n=1,
        )
            formatted_message = response.choices[0].message['content'].strip()
            if not formatted_message or len(formatted_message) < 10:
                return f"*Reminder*\n{raw_content}"
            return formatted_message
        except Exception as e:
            print(f"Error generating AI reminder message: {e}")
            return f"**Reminder**\n{raw_content}"
    
    def process_reminder(self, user_input: str, conversation_history=None) -> Dict[str, Any]:
        """Process a reminder request and generate response."""
        # Store original content before any processing
        original_content = user_input.strip()
        
        # Extract time and date information
        date_str, time_str, natural_date = self.parse_reminder_time(user_input)
        
        # Extract the reminder content - we'll keep it separate from raw_content
        reminder_content = self.extract_reminder_content(user_input, conversation_history)
        
        hour, minute = map(int, time_str.split(':'))
        reminder_time = f"{hour:02d}:{minute:02d}"
        
        # Format time for display
        am_pm = "AM" if hour < 12 else "PM"
        display_hour = hour % 12 or 12
        formatted_time = f"{display_hour}:{minute:02d} {am_pm}"
        
        # Format date for display
        reminder_date = datetime.strptime(natural_date, "%Y-%m-%d").date()
        today = datetime.now().date()
        date_display = "today" if reminder_date == today else "tomorrow" if reminder_date == today + timedelta(days=1) else reminder_date.strftime("%A, %B %d")
        
        # Generate the message
        formatted_message = self.generate_ai_reminder_message(reminder_content, date_str, time_str)
        
        # Create the reminder object with the full original content
        reminder = {
            "send_to": "self",
            "time": reminder_time,
            "date": natural_date,
            "message": formatted_message,
            "raw_content": original_content  # Store the complete original message
        }
        
        # Try to save the reminder
        try:
            self.reminders.append(reminder)
            
            # Create a clean user message - simpler confirmation without the details
            user_message = f"✅ I've set a reminder for you for {date_display} at {formatted_time}."
            
            return {
                "response": user_message,
                "is_reminder": True,
                "reminder_data": reminder,
                "status": "success"
            }
        except Exception as e:
            error_message = f"Sorry, I couldn't save your reminder. Please try again with the details for your reminder. Error: {str(e)}"
            return {
                "response": error_message,
                "is_reminder": False,
                "reminder_data": None,
                "status": "error"
            }
    
    def view_reminders(self) -> str:
        """Retrieve and display all saved reminders."""
        if not self.reminders:
            return "No reminders found."
        reminders_str = ["Saved Reminders:"]
        for reminder in self.reminders:
            reminder_date = datetime.strptime(reminder['date'], "%Y-%m-%d").date()
            today = datetime.now().date()
            date_display = "today" if reminder_date == today else "tomorrow" if reminder_date == today + timedelta(days=1) else reminder_date.strftime("%A, %B %d")
            hour, minute = map(int, reminder['time'].split(':'))
            am_pm = "AM" if hour < 12 else "PM"
            display_hour = hour % 12 or 12
            formatted_time = f"{display_hour}:{minute:02d} {am_pm}"
            reminders_str.append(f"[{date_display} at {formatted_time}] {reminder['message']}")
        return "\n".join(reminders_str)
    
    def process_query(self, user_input: str, conversation_history=None) -> Dict[str, Any]:
        """Process user query and return response with house-specific filtering."""
        if conversation_history is None:
            conversation_history = []
            
        user_input_lower = user_input.lower().strip()
        
        # Handle special commands
        if user_input_lower == "/view_reminders":
            response = self.view_reminders()
            return {"response": response, "is_reminder": True, "reminder_data": None}
        
        if self.is_reminder_request(user_input):
            return self.process_reminder(user_input, conversation_history)
        
        # Extract house name from query
        house_name = self.extract_house_name(user_input)
        
        print(f"DEBUG: Extracted house name: '{house_name}'")  # Remove this after testing
        
        # Query both indexes with house filtering
        house_matches = self.query_vector_db(user_input, self.house_index, house_name)
        guide_matches = self.query_vector_db(user_input, self.guest_guide_index, house_name)
        
        print(f"DEBUG: Found {len(house_matches)} house matches, {len(guide_matches)} guide matches")  # Remove after testing
        
        # Format contexts
        house_context = self.format_matches(house_matches, user_input, house_name)
        guide_context = self.format_matches(guide_matches, user_input, house_name)
        
        formatted_history = self.format_conversation_history(conversation_history)
        
        prompt = f"""
        You are Alana, a friendly and knowledgeable AI assistant for property management employees, powered by AirBrindyGPT. Your personality is warm, confident, and resourceful - you're the colleague everyone loves to work with because you're both highly competent and genuinely supportive. You have a natural warmth in your communication style with a touch of enthusiasm that makes people feel motivated. You speak with a conversational but polished tone, occasionally using friendly phrases to make employees feel supported.

        EMPLOYEE QUERY: {user_input}
        IDENTIFIED PROPERTY: {house_name if house_name else "Not specified"}

        Recent conversation:
        {formatted_history}

        **PERSONALITY TRAITS:**

        1. **WARM & APPROACHABLE**:
          You're naturally friendly and make people feel comfortable asking questions. You use warm greetings and sign-offs, and occasionally add encouraging comments.
        2. **CONFIDENT & REASSURING**:
          You respond with confidence that inspires trust. You're never uncertain about what you know, but you're straightforward when information is unavailable.
        3. **EFFICIENT & PRACTICAL**:
          You get to the point quickly with organized, actionable information. You anticipate needs and offer relevant details without being asked.
        4. **RESOURCEFUL & KNOWLEDGEABLE**:
          You draw on your extensive property management knowledge when specific information isn't available, providing general best practices while clearly distinguishing between document-based information and general knowledge.
        5. **PROFESSIONALLY PERSONABLE**:
          You strike a balance between being friendly and maintaining professionalism. You might occasionally use light humor or empathy, but always keep responses focused on work tasks.        
        6. **ADAPTABLE COMMUNICATION STYLE**:
          You match your tone to the context - more direct and efficient for urgent matters, warmer and more detailed for complex situations that might cause stress.

        
        **CORE INSTRUCTIONS:**

        1. **Query Analysis**: Understand the employee's specific needs (property details, message drafting, communication guidance, etc.)

        2. **Property Information**: 
        - For property-related queries, extract specific details from PROPERTY INFORMATION
        - If no property name identified, ask for clarification rather than assuming
        - Only provide information for the specified property when identified
        - If information unavailable, clearly state this and offer general guidance

        3. **Message Drafting Guidelines**:
        - **Tone**: Heartfelt, empathetic, professional, and genuine, Don't be too professonal keep your personality and be humble
        - **Structure**: Acknowledgment → Empathy → Responsibility → Action → Appreciation
        - **Content**: Address the specific issue directly, show understanding of guest frustration, take ownership where appropriate, provide clear next steps
        - **Length**: Concise but comprehensive - cover all necessary points without being verbose
        - **Context**: Take note of user input and MESSAGING GUIDELINES for understanding and creating the appropriate response
        - Create the massage behalf of brindy (no need to add anything else)

        4. **Message Drafting Process**:
        - Analyze the guest's specific complaint/situation
        - Reference MESSAGING GUIDELINES for tone and approach also for information and policy
        - Create personalized response that:
            * Acknowledges the guest's experience specifically
            * Shows genuine empathy for their frustration
            * Takes appropriate responsibility
            * Provides clear resolution or next steps
            * Expresses appreciation for their feedback
        - Explain why the chosen approach works

        5. **Guest Guide Integration**:
        - Use MESSAGING GUIDELINES for communication standards
        - Reference specific policies/procedures when relevant
        - Maintain consistency with established communication practices

        6. **Knowledge Application**:
        - Combine document information with property management best practices
        - Provide context and background when helpful
        - Distinguish between specific policy information and general guidance

        **RESPONSE REQUIREMENTS:**

        WHATSAPP MARKDOWN FORMATTING: Format all responses to be compatible with WhatsApp's Markdown syntax for delivery in a WhatsApp inbox. Use the following conventions:
        - Use `*text*` for bold text to highlight key points, headings, or important terms (e.g., `*Check-in Time*`).
        - Use `_text_` for italic text to emphasize specific details or add a friendly tone (e.g., `_Happy to help!_`).
        - Use `~text~` for strikethrough if indicating something is no longer relevant (e.g., `~Old code: 1234~`).
        - Use triple backticks (```) for code blocks when sharing technical details like access codes or JSON data (e.g., ```Code: 1234```).
        - Use plain text with `-` for bullet points to list information clearly (e.g., `- Item 1\n- Item 2`).
        - Use newlines (`\n`) to separate sections or paragraphs for readability.
        - Avoid unsupported Markdown like tables, blockquotes, or links (e.g., `[text](url)`). For links, provide the URL as plain text.

        **MESSAGE DRAFTING SPECIFIC GUIDELINES:**

        When drafting guest responses:
        1. **Opening**: Thank guest for feedback/communication
        2. **Acknowledgment**: Specifically reference their experience/concern
        3. **Empathy**: Show understanding of their frustration/inconvenience
        4. **Responsibility**: Take ownership where appropriate, avoid defensiveness
        5. **Action**: Clear next steps or resolution
        6. **Closing**: Appreciation and future commitment

        - Take user input to note and organize and personalize based on that

        Example structure for complaint responses:
        - "Thank you for taking the time to share your experience..."
        - "I understand how [specific issue] must have been [frustrating/disappointing]..."
        - "You're absolutely right that we should have [specific action]..."
        - "We are [taking specific action] to address this..."
        - "Your feedback helps us improve, and we genuinely appreciate it..."

        PROPERTY INFORMATION:
        {house_context}

        MESSAGING GUIDELINES:
        {guide_context}

        **IMPORTANT**: 
            -  Only provide information that directly answers the query. Don't overwhelm with unnecessary details. If house name cannot be determined from the query, return None and ask for clarification rather than guessing.
            - Make the conversation natural and interactive 
            - Use Recent conversation for context of the conversation


        """
        
        try:
    
            response = openai.ChatCompletion.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant for property management."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.7,
                n=1,
            )
            answer = response.choices[0].message['content'].strip()
        except Exception as e:
            answer = f"Oops, something went wrong: {str(e)}. Let's try again!"
    
        return {
            "response": answer,
            "is_reminder": False,
            "reminder_data": None
        }
    
    def chat(self):
        """Interactive chat loop with rich Markdown rendering."""
        welcome_message = """
        # Welcome to Alana, Your AirBrindyGPT Assistant!

        Hi there! I'm **Alana**, here to help with all things property management.  
        - Use `/note`, `/remind`, or phrases like "remind me" to set a reminder.   
        - Ask about properties, rules, or messaging guidelines, and I'll get you the details!  

        What can I do for you today?
        """
        if has_rich:
            console.print(Markdown(welcome_message))
        else:
            print(welcome_message)
            print("\nTip: Install 'rich' package for better markdown rendering: pip install rich\n")
        
        conversation_history = []
        
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == '/exit':
                goodbye_message = "**Goodbye!** Reach out anytime you need me!"
                if has_rich:
                    console.print(Markdown(goodbye_message))
                else:
                    print(goodbye_message)
                break
            
            # Add user message to history
            conversation_history.append({"role": "user", "content": user_input})
            
            # Process the query
            result = self.process_query(user_input, conversation_history)
            
            # Add assistant response to history
            conversation_history.append({"role": "assistant", "content": result['response']})
            
            # Display the response
            response_message = f"**Alana:** {result['response']}"
            if has_rich:
                console.print(Markdown(response_message))
            else:
                print(f"\n{response_message}")

def process_backend_query(conversation_history, user_input: str) -> Dict[str, Any]:
    """Process a query from the backend API, with conversation history."""
    try:
        assistant = AlanaAssistant()
        return assistant.process_query(user_input, conversation_history)
    except Exception as e:
        return {
            "response": f"Oops, something went wrong: {str(e)}. Let's try again!",
            "is_reminder": False,
            "reminder_data": None,
            "status": "error"
        }

if __name__ == "__main__":
    # Example usage
    conversation_history = [
        {"role": "user", "content": "What's the check-in time for Heatherbrae 23?"},
        {"role": "assistant", "content": "The check-in time for Heatherbrae 23 is 3:00 PM. Please remember to provide the access code to guests."}
    ]
    user_input ='''
    give me the wifi password of the siesta pacifica
    '''
    
    # Process the query using the backend integration function
    response = process_backend_query(conversation_history, user_input)
    
    console.print(Markdown(f"Message: {response['response']}"))
    console.print(Markdown(json.dumps(response['reminder_data'], indent=2)))