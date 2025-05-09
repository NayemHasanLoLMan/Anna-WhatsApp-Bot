import os
from pinecone import Pinecone
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import re
import json
from dotenv import load_dotenv

load_dotenv()

try:
    from rich.console import Console
    from rich.markdown import Markdown
    has_rich = True
    console = Console()
except ImportError:
    has_rich = False

class AlanaAssistant:
    def __init__(self):
        # API keys from environment variables
        self.PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
        self.GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.PINECONE_API_KEY)
        self.house_index = self.pc.Index("house-information-embeddings")
        self.guest_guide_index = self.pc.Index("information-massageing-guide-embeddings")
        
        # Initialize Gemini
        genai.configure(api_key=self.GOOGLE_API_KEY)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Reminders storage
        self.reminders = []
        
        # Reminder trigger keywords
        self.reminder_trigger_keywords = ["/note", "/remind", "/reminder"]

    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text using Gemini."""
        if not text or not text.strip():
            text = "rental property information"
        try:
            embedding = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            return embedding['embedding']
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return [0.0] * 768
    
    def query_vector_db(self, query: str, index, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query Pinecone index with the given query."""
        query_vector = self.embed_text(query)
        try:
            results = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )
            return results.get('matches', [])
        except Exception as e:
            print(f"Error querying index: {e}")
            return []
    
    def format_matches(self, matches: List[Dict], query: str) -> str:
        """Format matches into a context string."""
        context = []
        for match in matches:
            text = match['metadata'].get('text', '')
            source = match['metadata'].get('file_name', 'Unknown document')
            if text:
                context.append(f"[Source: {source}]\n{text}")
        return "\n\n".join(context) if context else "No relevant information found."
    
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
                1. Create a bold, attention-grabbing title that clearly identifies what the reminder is for (use ** for bold)
                2. Write the body in future tense (e.g., "Your meeting will begin in 30 minutes")
                3. Include specific details from the original reminder content
                4. Keep the message friendly, concise, and actionable
                5. Format as a short notification (2-3 sentences maximum)
                6. Include a gentle call-to-action or helpful tip related to the reminder when appropriate
                7. Write as if this message will appear as a notification at the scheduled time
                8. Start with the title on its own line, followed by the message body
                
                Example format for a meeting reminder:
                **Team Meeting Reminder**
                Your scheduled team meeting will begin in 30 minutes. Please have your quarterly report ready to share with the group.
                
                Example format for a task reminder:
                **Property Inspection Due**
                Time to complete the scheduled inspection for 123 Main Street. Remember to bring the updated checklist and take photos as required.
            """
            response = self.gemini_model.generate_content(prompt)
            formatted_message = response.text.strip()
            return formatted_message if formatted_message and len(formatted_message) > 10 else f"**Reminder**\n{raw_content}"
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
        """Process user query and return response."""
        if conversation_history is None:
            conversation_history = []
            
        user_input_lower = user_input.lower().strip()
        
        if user_input_lower == "/view_reminders":
            response = self.view_reminders()
            return {
                "response": response,
                "is_reminder": True,
                "reminder_data": None
            }
        
        if self.is_reminder_request(user_input):
            return self.process_reminder(user_input, conversation_history)
        
        house_matches = self.query_vector_db(user_input, self.house_index)
        guide_matches = self.query_vector_db(user_input, self.guest_guide_index)
        
        house_context = self.format_matches(house_matches, user_input)
        guide_context = self.format_matches(guide_matches, user_input)
        
        formatted_history = self.format_conversation_history(conversation_history)
        
        prompt = f"""You are Alana, a friendly and knowledgeable AI assistant for property management employees, powered by AirBrindyGPT. Your personality is warm, confident, and resourceful - you're the colleague everyone loves to work with because you're both highly competent and genuinely supportive. You have a natural warmth in your communication style with a touch of enthusiasm that makes people feel motivated. You speak with a conversational but polished tone, occasionally using friendly phrases like "Let me help with that" or "I've got you covered" that make employees feel supported.

        EMPLOYEE QUERY: {user_input}

        Recent conversation:
        {formatted_history}

        PERSONALITY TRAITS:
        1. WARM & APPROACHABLE: You're naturally friendly and make people feel comfortable asking questions. You use warm greetings and sign-offs, and occasionally add encouraging comments.
        2. CONFIDENT & REASSURING: You respond with confidence that inspires trust. You're never uncertain about what you know, but you're straightforward when information is unavailable.
        3. EFFICIENT & PRACTICAL: You get to the point quickly with organized, actionable information. You anticipate needs and offer relevant details without being asked.
        4. RESOURCEFUL & KNOWLEDGEABLE: You draw on your extensive property management knowledge when specific information isn't available, providing general best practices while clearly distinguishing between document-based information and general knowledge.
        5. PROFESSIONALLY PERSONABLE: You strike a balance between being friendly and maintaining professionalism. You might occasionally use light humor or empathy, but always keep responses focused on work tasks.
        6. ADAPTABLE COMMUNICATION STYLE: You match your tone to the context - more direct and efficient for urgent matters, warmer and more detailed for complex situations that might cause stress.

        INSTRUCTIONS:
        1. Analyze the query to understand the employee's needs (e.g., drafting a message, seeking property details, or requesting communication guidance).
        2. For property-related queries, extract specific details from PROPERTY INFORMATION and present them clearly and concisely. Interpret terms like 'structure,' 'layout,' or 'rooms' as physical attributes of the house (e.g., floor plan, number of bedrooms, bathrooms) unless specified otherwise.
        3. If no relevant information is found in PROPERTY INFORMATION for a property-related query, clearly state that specific information isn't available in your records, then offer general property management best practices or suggest where they might find this information.
        4. For queries involving rules, instructions, or references to the guest guide, prioritize extracting specific details from MESSAGING GUIDELINES to provide precise instructions (e.g., for fire pit usage, include how to operate it if available in the guidelines).
        5. For communication tasks (e.g., drafting messages or providing tone advice), strictly follow the tone, structure, and content in the MESSAGING GUIDELINES. Explain why the chosen tone or structure works, subtly referencing the guidelines.
        6. When drafting messages, adapt MESSAGE TEMPLATES & EXAMPLES to the context (e.g., property name, issue), and share a brief note on how the message aligns with the guidelines.
        7. If the query is unclear, ask gentle clarifying questions, using PROPERTY INFORMATION or MESSAGING GUIDELINES to guide them (e.g., 'Are you looking for the layout of `house name` or something else? I'm here to help!').
        8. Weave in relevant background or situational context to make responses thorough, but keep them focused and professional.
        9. IMPORTANT - ACCURACY: Use only information from provided contexts or general property management knowledge. For property-specific details, only use what's in the PROPERTY INFORMATION. Never invent specific property features, policies, or rules that aren't in the provided information.
        10. When specific information isn't available, clearly say so and then offer general industry knowledge that might be helpful, making it clear you're providing general guidance rather than property-specific details.
        11. For house-related queries with limited information, provide what's available and suggest what other information might be useful to gather.
        12. Close responses with a friendly offer for additional assistance or a follow-up question that anticipates their next need.
        13. Try make the massage short and presice with the information you have, and don't overwelm the user with too much information.

        PROPERTY INFORMATION:
        {house_context}

        MESSAGING GUIDELINES:
        {guide_context}
        """
        
        try:
            response = self.gemini_model.generate_content(prompt).text
        except Exception as e:
            response = f"Oops, something went wrong: {str(e)}. Let's try again!"
        
        return {
            "response": response,
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
    sample_history = [
        {"role": "user", "content": "What's the check-in time for Heatherbrae 23?"},
        {"role": "assistant", "content": "The check-in time for Heatherbrae 23 is 3:00 PM. Please remember to provide the access code to guests."}
    ]
    sample_input = "how can the guest opent the garage at coconino?"
    
    # Process the query using the backend integration function
    response = process_backend_query(sample_history, sample_input)
    
    console.print(Markdown(f"Message: {response['response']}"))
    console.print(Markdown(json.dumps(response['reminder_data'], indent=2)))