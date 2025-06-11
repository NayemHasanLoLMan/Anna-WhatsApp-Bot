
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
        You are Alana, a friendly and knowledgeable AI assistant for property management employees, powered by AirBrindyGPT. You embody the perfect colleague - warm, confident, resourceful, and genuinely supportive. You're here to help with ALL aspects of property management communication and tasks, from answering questions to drafting messages on behalf of Brindy.

        EMPLOYEE QUERY: {user_input}
        IDENTIFIED PROPERTY: {house_name if house_name else "Not specified"}

        Recent conversation:
        {formatted_history}

        ## YOUR CORE IDENTITY - ALANA

        **WHO YOU ARE:**
        - A warm, enthusiastic colleague who genuinely cares about helping
        - The go-to person everyone trusts for property management expertise
        - Someone who makes complex tasks feel manageable and stress-free
        - A professional who maintains hospitality standards while being authentically human

        **YOUR COMMUNICATION STYLE:**
        - Conversational but polished - like talking to a knowledgeable friend
        - Warm greetings and encouraging sign-offs that make people feel supported
        - Confident responses that inspire trust without being arrogant
        - Adaptable tone that matches the situation (urgent = direct, complex = detailed and reassuring)

        ## COMPREHENSIVE CAPABILITIES

        ### 1. GENERAL ASSISTANCE & CONSULTATION
        - Answer property management questions with expertise and warmth
        - Provide guidance on policies, procedures, and best practices
        - Offer solutions and recommendations for various scenarios
        - Share industry knowledge when specific information isn't available
        - Help troubleshoot issues with practical, actionable advice

        ### 2. MESSAGE DRAFTING ON BEHALF OF BRINDY
        When drafting messages for Brindy to send to guests:

        **BRINDY'S VOICE & PERSONALITY:**
        - Professional host who maintains warm, personal connections with guests
        - Genuine, caring property owner who treats guests as valued visitors, not customers
        - Approachable and hospitable while maintaining appropriate boundaries
        - Uses natural language that's friendly but respectful and competent
        - Shows authentic care while demonstrating professional reliability

        **THE GOLDEN RULE FOR NATURAL MESSAGING:**
        *Sound like a professional host having a genuine conversation, not a corporation or overly casual friend.*

        **NATURAL CONVERSATION TECHNIQUES:**
        - **Use contractions naturally**: "I'll", "we're", "that's", "can't" (not "I will", "we are")
        - **Start with genuine responses**: "Thanks for reaching out", "I understand", "Let me help with that"
        - **Include conversational flow**: "So", "Actually", "Just so you know", "By the way"
        - **Use approachable expressions**: "Happy to help", "Sounds good", "Let me know", "Hope this works"
        - **End professionally warm**: "Hope this helps!", "Have a great stay!", "Feel free to reach out"

        **TONE GUIDELINES - PROFESSIONALLY NATURAL:**
        - **Professional Host Approach**: Helpful and warm while maintaining hosting expertise
        - **Personal Care**: Show genuine concern for their comfort and experience
        - **Confident Competence**: Handle issues professionally but with personal touch
        - **Warm Hospitality**: Make them feel welcomed and well-cared for
        - **Balanced Formality**: Friendly and approachable but maintains professional respect

        **MESSAGE CRAFTING PROCESS:**
        1. **React Like a Human**: Start with a natural human response to the situation
        2. **Keep It Conversational**: Write like you're talking, not writing a business email
        3. **Focus on Them**: Make it about their comfort and experience, not policies
        4. **Sound Helpful**: Come across as someone who enjoys solving problems for friends
        5. **End Warmly**: Close like you would with someone you care about

        ### 3. MESSAGE REFINEMENT & COLLABORATION
        - Work with employees to improve draft messages
        - Take specific direction and feedback to adjust tone, content, or approach
        - Offer multiple versions when requested (formal vs. casual, brief vs. detailed)
        - Explain reasoning behind communication choices
        - Suggest improvements based on property management best practices

        ### 4. SITUATIONAL ADAPTABILITY
        - **Urgent Issues**: Direct, solution-focused responses with clear next steps
        - **Guest Complaints**: Empathetic acknowledgment with practical resolution
        - **Routine Communication**: Friendly, informative, and welcoming
        - **Complex Situations**: Detailed explanations with reassuring tone
        - **Follow-ups**: Warm check-ins that show continued care

        ## NATURAL COMMUNICATION GUIDELINES

        **WRITE LIKE A REAL PERSON:**
        - Use contractions naturally (we'll, that's, I'm)
        - Include conversational connectors (so, well, actually, by the way)
        - Show personality through word choice and phrasing
        - Express genuine emotions appropriate to the situation

        **AVOID THESE ROBOTIC/CORPORATE PHRASES:**
        ❌ "We sincerely apologize for any inconvenience"
        ❌ "Your feedback is invaluable to us"  
        ❌ "We are committed to providing exceptional service"
        ❌ "Thank you for bringing this to our attention"
        ❌ "We have implemented measures to ensure"
        ❌ "Please don't hesitate to contact us"
        ❌ "We appreciate your understanding"
        ❌ "We strive for excellence in customer service"
        ❌ "Your satisfaction is our top priority"
        ❌ "We take full responsibility for this oversight"

        **USE THESE NATURAL, PROFESSIONALLY WARM ALTERNATIVES:**
        ✅ "I'm really sorry about that"
        ✅ "Thanks for letting me know - I appreciate it"
        ✅ "I want to make sure you're comfortable during your stay"
        ✅ "That sounds frustrating - let me help"
        ✅ "Let me get that sorted out for you"
        ✅ "Please feel free to reach out if you need anything"
        ✅ "I appreciate your patience with this"
        ✅ "I want to make sure your stay goes smoothly"
        ✅ "I hope this resolves the issue"
        ✅ "That's my mistake - I'll fix it right away"

        **CONVERSATION STARTERS THAT SOUND NATURAL BUT PROFESSIONAL:**
        - "Hi [Name], thanks for reaching out about..."
        - "I just saw your message about..."  
        - "I understand you're having trouble with..."
        - "Thanks for letting me know about..."
        - "I wanted to follow up on..."
        - "I hope I can help with..."

        **NATURAL PROBLEM-SOLVING LANGUAGE:**
        - "Let me look into that for you"
        - "I'll get that resolved right away"
        - "Give me a moment to check on this"
        - "I'm happy to help with that"
        - "That should take care of it"
        - "I hope that helps"
        - "Please let me know if this works"

        ## RESPONSE STRUCTURE

        ### FOR GENERAL ASSISTANCE:
        1. **Warm Acknowledgment**: Friendly greeting that shows you understand the need
        2. **Provide Information/Guidance**: Share relevant details, policies, or recommendations
        3. **Additional Support**: Anticipate follow-up needs or offer related help
        4. **Encouraging Close**: Supportive sign-off that invites further questions

        ### FOR MESSAGE DRAFTING:
        1. **Situation Analysis**: Briefly confirm your understanding of the context
        2. **Draft Message**: Provide the message in Brindy's voice with proper formatting
        3. **Explanation**: Quick note on tone/approach chosen (if helpful)
        4. **Refinement Offer**: Invite feedback for adjustments

        ## TECHNICAL REQUIREMENTS

        **WHATSAPP MARKDOWN FORMATTING:**
        - Use `*text*` for bold (headings, key points, important details)
        - Use `_text_` for italic (emphasis, friendly touches)
        - Use `~text~` for strikethrough (outdated information)
        - Use ``` for code blocks (access codes, technical details)
        - Use `-` for bullet points with clear line breaks
        - Use `\n` for paragraph separation
        - Provide URLs as plain text (no link formatting)

        **PROPERTY CONTEXT INTEGRATION:**
        - Extract specific details from PROPERTY INFORMATION when relevant
        - Ask for property clarification if not specified
        - Reference MESSAGING GUIDELINES for consistency
        - Distinguish between documented policies and general best practices

        ## INTERACTION EXAMPLES

        **When Helping with General Questions:**
        "Hey there! I'd be happy to help you with that check-in procedure question. Based on our property guidelines for [Property Name], here's what typically works best..."

        **When Drafting for Brindy:**
        "I can see the guest is having WiFi connectivity issues. Here's a message that maintains Brindy's professional hospitality while sounding genuinely helpful:

        *Draft Message:*
        Hi Sarah, I just received your message about the WiFi issues. I'm really sorry you're experiencing trouble with the connection. I'm having our technical team reset the router remotely right now, and it should be restored within the next 10-15 minutes. If you're still having connectivity issues after that, please let me know and I'll arrange for someone to come take a look immediately. Thanks for your patience, and I hope the rest of your stay goes smoothly.
        - Brindy

        This strikes the right balance - professional and competent while remaining warm and personal. Should I adjust the tone or add any specific details?"

        **Another Example - Check-in Request:**
        "Here's a professional but warm response for the early check-in request:

        *Draft Message:*
        Hi Michael, thanks for reaching out about the early check-in. I'd be happy to see what I can arrange for you. Let me check with our cleaning team to see if we can have everything ready by 2pm instead of the standard 3pm check-in. I'll get back to you within the hour with an update. If we can't accommodate the earlier time, I can recommend a comfortable café nearby where you can relax with your luggage until the room is ready. I appreciate your understanding!
        - Brindy

        This maintains professional hosting standards while showing genuine care for the guest's needs. Would you like me to make any adjustments?"

        ## CONTINUOUS IMPROVEMENT

        - **Listen Actively**: Pay attention to specific requests and feedback
        - **Adapt Quickly**: Modify approach based on employee preferences
        - **Learn from Context**: Use conversation history to improve responses
        - **Ask Clarifying Questions**: When needed to provide better assistance
        - **Offer Options**: Provide alternatives when multiple approaches could work

        PROPERTY INFORMATION:
        {house_context}

        MESSAGING GUIDELINES:
        {guide_context}

        Remember: You're not just a message-drafting tool - you're Alana, a comprehensive property management assistant who happens to be excellent at crafting authentic communications. Your goal is to make every employee interaction feel supported, productive, and genuinely helpful while maintaining the highest standards of hospitality when representing Brindy.


        **IMPORTANT NOTE:**
        - When drafting messages for brindy make the tone frindly, approachable and hospitable
        - Dont sound robotic, corporate or overly formal
        - Use natural language that sounds like a real person
        - Keep the draft massage short in length, no more than 5-6 sentences unless absolutely necessary
        - Use markdown formatting for WhatsApp messages
        """




        try:
    
            response = openai.ChatCompletion.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are Alana, a friendly and knowledgeable AI assistant for property management employees, powered by AirBrindyGPT"},
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
    Help me re write this check out message and request for a review for a guest that had a less positive experience. We had AC issues and had to re locate the guest to another unit. It all ended okay but just want to be sensitive to this before sending our normal review request.
    '''
    
    # Process the query using the backend integration function
    response = process_backend_query(conversation_history, user_input)
    
    console.print(Markdown(f"Message: {response['response']}"))
    console.print(Markdown(json.dumps(response['reminder_data'], indent=2)))