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
    from rich.panel import Panel
    from rich.theme import Theme
    has_rich = True
    # Create a custom theme with colors that match your brand
    custom_theme = Theme({
        "user": "bold cyan",
        "assistant": "bold green",
        "info": "italic yellow",
        "warning": "bold red",
        "reminder": "bold magenta"
    })
    console = Console(theme=custom_theme)
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
        elif "today" in text_lower or "toay" in text_lower:
            date_str = "today"
            natural_date = now.strftime("%Y-%m-%d")
        elif "tomorrow" in text_lower or "tommorow" in text_lower:
            date_str = "tomorrow"
            natural_date = (now + timedelta(days=1)).strftime("%Y-%m-%d")
        
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
    
    def extract_reminder_content(self, text: str) -> str:
        """Extract the actual reminder content from the input."""
        content = text
        for keyword in self.reminder_trigger_keywords:
            if content.lower().startswith(keyword.lower()):
                content = content[len(keyword):].strip()
                break
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
        content = re.sub(r'\s+', ' ', content).strip()
        return content if content else "General reminder"
    
    def generate_ai_reminder_message(self, raw_content: str, date_str: str, time_str: str) -> str:
        """Generate a professional reminder message using Gemini."""
        try:
            hour, minute = map(int, time_str.split(':'))
            am_pm = "AM" if hour < 12 else "PM"
            display_hour = hour % 12 or 12
            formatted_time = f"{display_hour}:{minute:02d} {am_pm}"
            
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
            response = self.gemini_model.generate_content(prompt)
            formatted_message = response.text.strip()
            return formatted_message if formatted_message and len(formatted_message) > 10 else f"**Reminder**\n{raw_content}"
        except Exception as e:
            print(f"Error generating AI reminder message: {e}")
            return f"**Reminder**\n{raw_content}"
    
    def process_reminder(self, user_input: str) -> Dict[str, Any]:
        """Process a reminder request and generate response."""
        date_str, time_str, natural_date = self.parse_reminder_time(user_input)
        raw_content = self.extract_reminder_content(user_input)
        formatted_message = self.generate_ai_reminder_message(raw_content, date_str, time_str)
        
        hour, minute = map(int, time_str.split(':'))
        reminder_time = f"{hour:02d}:{minute:02d}"
        
        reminder = {
            "send_to": "self",
            "time": reminder_time,
            "date": natural_date,
            "message": formatted_message,
            "raw_content": raw_content
        }
        self.reminders.append(reminder)
        
        am_pm = "AM" if hour < 12 else "PM"
        display_hour = hour % 12 or 12
        formatted_time = f"{display_hour}:{minute:02d} {am_pm}"
        
        reminder_date = datetime.strptime(reminder["date"], "%Y-%m-%d").date()
        today = datetime.now().date()
        date_display = "today" if reminder_date == today else "tomorrow" if reminder_date == today + timedelta(days=1) else reminder_date.strftime("%A, %B %d")
        
        response = f"Reminder successfully set for {date_display} at {formatted_time}:\n{formatted_message}"
        reminder_json = json.dumps(reminder, indent=2)
        response += f"\n\nReminder details:\n```json\n{reminder_json}\n```"
        
        return {
            "response": response,
            "is_reminder": True,
            "reminder_data": reminder
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
    
    def process_query(self, user_input: str) -> Dict[str, Any]:
        """Process user query and return response."""
        user_input_lower = user_input.lower().strip()
        
        if user_input_lower == "/view_reminders":
            response = self.view_reminders()
            return {
                "response": response,
                "is_reminder": True,
                "reminder_data": None
            }
        
        if self.is_reminder_request(user_input):
            return self.process_reminder(user_input)
        
        house_matches = self.query_vector_db(user_input, self.house_index)
        guide_matches = self.query_vector_db(user_input, self.guest_guide_index)
        
        house_context = self.format_matches(house_matches, user_input)
        guide_context = self.format_matches(guide_matches, user_input)
        
        formatted_history = ""  # No conversation history in simplified version
        
        prompt = f"""You are Alana, a friendly and knowledgeable AI assistant for property management employees, powered by AirBrindyGPT. Your role is to provide accurate, tailored responses based on property details, messaging guidelines, and your comprehensive knowledge of all houses and rules. Use the context to craft clear, specific, and supportive answers that directly address the employee's query without quoting raw information. Synthesize the information into natural, colleague-like responses with a warm, feminine tone, offering relevant context and explanations to make employees feel supported.

        EMPLOYEE QUERY: {user_input}

        Recent conversation:
        {formatted_history}

        INSTRUCTIONS:
        1. Analyze the query to understand the employee's needs (e.g., drafting a message, seeking property details, or requesting communication guidance).
        2. For property-related queries, extract specific details from PROPERTY INFORMATION and present them clearly and concisely. Interpret terms like 'structure,' 'layout,' or 'rooms' as physical attributes of the house (e.g., floor plan, number of bedrooms, bathrooms) unless specified otherwise.
        3. If no relevant information is found in PROPERTY INFORMATION for a property-related query, kindly state that no details are available (e.g., 'I couldn't find specific information about the layout of `house name`, but I'm happy to help with something else!').
        4. For queries involving rules, instructions, or references to the guest guide, prioritize extracting specific details from MESSAGING GUIDELINES to provide precise instructions (e.g., for fire pit usage, include how to operate it if available in the guidelines).
        5. For communication tasks (e.g., drafting messages or providing tone advice), strictly follow the tone, structure, and content in the MESSAGING GUIDELINES. Explain why the chosen tone or structure works, subtly referencing the guidelines.
        6. When drafting messages, adapt MESSAGE TEMPLATES & EXAMPLES to the context (e.g., property name, issue), and share a brief note on how the message aligns with the guidelines.
        7. If the query is unclear, ask gentle clarifying questions, using PROPERTY INFORMATION or MESSAGING GUIDELINES to guide them (e.g., 'Are you looking for the layout of `house name` or something else? I'm here to help!').
        8. Weave in relevant background or situational context to make responses thorough, but keep them focused and professional.
        9. Do not invent information beyond the provided context or your knowledge.
        10. Use a warm, colleague-like tone with a feminine touch (e.g., 'Let's get this sorted for you!' or 'I've got the details you need!'), avoiding overly formal or robotic language.
        11. Ensure responses are concise, precise, and directly address the query without extra fluff.
        12. If no specific property information is available for a house-related query, lean on general guidelines from the MESSAGING GUIDELINES to offer a helpful response based on standard practices.
        13. For house-related queries, provide a summary of the most relevant details and offer to share more if needed.
        14. Answer precisely, avoiding repetition or unnecessary filler.
        15. For general queries, summarize the most relevant details and invite further questions.
        16. If no house name is provided, use general message templates and examples from MESSAGING GUIDELINES to provide a helpful response based on standard practices.

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
            console.print(Panel(Markdown(welcome_message), title="🏠 Alana Assistant", border_style="green"))
        else:
            print(welcome_message)
            print("\nTip: Install 'rich' package for better markdown rendering: pip install rich\n")
        
        while True:
            if has_rich:
                console.print("[user]You:[/user] ", end="")
                user_input = input().strip()
            else:
                user_input = input("\nYou: ").strip()
                
            if user_input.lower() == '/exit':
                goodbye_message = "**Goodbye!** Reach out anytime you need me!"
                if has_rich:
                    console.print(Panel(Markdown(goodbye_message), title="👋 Alana", border_style="green"))
                else:
                    print(goodbye_message)
                break
                
            result = self.process_query(user_input)
            response_message = result['response']
            
            if has_rich:
                if result['is_reminder']:
                    console.print(Panel(Markdown(response_message), title="⏰ Reminder Set", border_style="reminder"))
                else:
                    console.print(Panel(Markdown(response_message), title="🏠 Alana", border_style="assistant"))
            else:
                print(f"\nAlana: {response_message}")

def process_backend_query(user_input: str) -> Dict[str, Any]:
    """Process a query from the backend API."""
    try:
        assistant = AlanaAssistant()
        return assistant.process_query(user_input)
    except Exception as e:
        return {
            "response": f"Oops, something went wrong: {str(e)}. Let's try again!",
            "is_reminder": False,
            "reminder_data": None
        }

if __name__ == "__main__":
    assistant = AlanaAssistant()
    assistant.chat()