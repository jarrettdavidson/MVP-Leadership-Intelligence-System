import os
import pickle
import hashlib
import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from dateutil.parser import parse as parse_date
from openai import OpenAI
from _client import get_lifelogs  # Your existing Limitless lifelog fetcher

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pytz

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('leadership_intelligence.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
CST = pytz.timezone("America/Winnipeg")
SCOPES = [
    'https://www.googleapis.com/auth/calendar.readonly',
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive.readonly',
    'https://www.googleapis.com/auth/documents.readonly',
]

# Configuration
class Config:
    """Configuration class for the Leadership Intelligence System"""
    
    def __init__(self):
        self.credentials_path = r"G:\My Drive\Jarrett Folders\JD INC\Limitle3ss\enhanced limitless\credentials.json"
        self.output_dir = r"G:\My Drive\Jarrett Folders\JD INC\Limitle3ss\MVP-Leadership Intelligence\output"
        self.spreadsheet_id = "1PHJhfYdkjJ1zR_3ruapii2OS3FeZYdeipsdZh8kUrCw"
        self.calendar_ids = [
            'primary',
            'piq3g614i5rvndfk01jtnj7gi1i0ttnj@import.calendar.google.com',
            'jarrettim@gmail.com'
        ]
        self.max_tasks_to_display = 10
        self.max_emails_per_category = 5
        self.task_similarity_threshold = 0.6
        
        # Validate environment variables
        self._validate_env_vars()
    
    def _validate_env_vars(self) -> None:
        """Validate that required environment variables are set"""
        required_vars = ["OPENAI_API_KEY", "LIMITLESS_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

# Prompt Templates
class PromptTemplates:
    """Centralized prompt templates for AI interactions"""
    
    DETAILED_LIFELOG_SUMMARY = (
        "You are Jarrett's executive assistant analyzing conversations from the past few days. "
        "Your goal is to extract specific commitments and promises he made to named individuals.\n\n"
        "ANALYSIS REQUIREMENTS:\n"
        "1. Extract every commitment or promise made to specific people (include their names)\n"
        "2. Include FULL CONTEXT for each commitment - what is being promised, why, and to whom\n"
        "3. Identify key wins and accomplishments from all conversations\n"
        "4. Suggest one improvement for upcoming days\n"
        "5. Note any contradictions or inconsistencies across conversations\n"
        "6. Create a prioritized action list with dates/times when mentioned\n"
        "7. Pay attention to conversation dates and organize findings chronologically\n"
        "8. When you see phrases like 'I'll find out' or 'I'll do that', include enough context to understand WHAT is being referenced\n\n"
        "FORMAT YOUR RESPONSE AS:\n"
        "🎯 COMMITMENTS & PROMISES:\n"
        "• [Specific commitment with full context] → [Person's name] → [Deadline if mentioned] → [Date of conversation]\n\n"
        "🏆 KEY WINS:\n"
        "• [Achievement or positive outcome] → [Date if relevant]\n\n"
        "💡 IMPROVEMENT SUGGESTION:\n"
        "• [One specific actionable suggestion for upcoming days]\n\n"
        "⚠️ CONTRADICTIONS/INCONSISTENCIES:\n"
        "• [Any contradictory statements across different conversations]\n\n"
        "📋 HIGH-PRIORITY ACTION ITEMS:\n"
        "• [Action item with context] - [Due date/time if available] - [Source conversation date]\n\n"
        "CRITICAL: Always provide enough context so commitments are actionable. 'I'll find out' should become 'I'll find out about [specific topic]' based on the conversation context."
    )
    
    TASK_EXTRACTION = (
        "You are an expert task extraction system for Jarrett's leadership intelligence.\n\n"
        "EXTRACTION CRITERIA:\n"
        "- Extract only specific commitments to named individuals\n"
        "- Avoid duplicating existing tasks\n"
        "- Focus on actionable items with clear outcomes\n"
        "- Include context from conversations, emails, or calendar events\n\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY a valid JSON array with this structure:\n"
        "[\n"
        "  {\n"
        "    \"task\": \"Specific action with person's name\",\n"
        "    \"priority\": \"P1/P2/P3\",\n"
        "    \"area\": \"T&T/Wedding/Personal/Business\",\n"
        "    \"due_date\": \"YYYY-MM-DD or 'Today'\",\n"
        "    \"notes\": \"Context from conversation/email/calendar\"\n"
        "  }\n"
        "]\n\n"
        "If no actionable tasks are found, return an empty array: []"
    )
    
    LEADERSHIP_SUMMARY = (
        "You are Jarrett's executive assistant creating his daily leadership intelligence briefing.\n\n"
        "Generate a comprehensive summary using this EXACT format:\n\n"
        "🎯 JARRETT'S LEADERSHIP INTELLIGENCE SUMMARY\n"
        "================================================================================\n"
        "📅 Analysis Date: {date_str}\n"
        "🕐 Generated: {time_str}\n"
        "📊 Data Sources: {total_conversations} conversations + {total_emails} emails + {total_events} events\n\n"
        "📅 TODAY'S CALENDAR\n"
        "----------------------------------------\n"
        "[List today's events with times and key attendees]\n\n"
        "📋 HIGH-PRIORITY TASKS (from ToDo Task System)\n"
        "----------------------------------------\n"
        "[P1 and P2 tasks from task management system with due dates]\n\n"
        "📧 EMAIL ACTION ITEMS\n"
        "----------------------------------------\n"
        "[Priority items requiring response or action from emails]\n\n"
        "📅 TO BE SCHEDULED IN CALENDAR\n"
        "----------------------------------------\n"
        "[Commitments from conversations that mention specific times/dates like 'on this day', 'at this time', 'next week', 'by tomorrow', 'this Friday', etc. Format: • [Commitment] → [Person] → [Time reference]]\n\n"
        "📝 TO BE ADDED TO TODO TASKS\n"
        "----------------------------------------\n"
        "[Commitments from conversations with phrases like 'I will', 'I should', 'I might', 'I need to', 'I'll follow up', etc. Format: • [Commitment] → [Person] → [Context]]\n\n"
        "👥 PEOPLE TO FOLLOW UP WITH\n"
        "----------------------------------------\n"
        "[Names and topics requiring follow-up with source reference]\n\n"
        "🧠 LEADERSHIP INSIGHT\n"
        "----------------------------------------\n"
        "[One actionable insight for leadership, fatherhood, or fitness improvement]\n\n"
        "📝 MEETING PREPARATION\n"
        "----------------------------------------\n"
        "[Specific prep items for upcoming meetings]\n\n"
        "================================================================================\n"
        "✅ Leadership Intelligence Summary Complete\n\n"
        "IMPORTANT: \n"
        "- Extract real, specific information from the provided data\n"
        "- Use actual names, dates, and details\n"
        "- Split commitments by time-based vs action-based language\n"
        "- Avoid duplicating items between calendar and todo sections\n"
        "- Focus on actionable intelligence that helps Jarrett be more effective"
    )

class GoogleCredentialsManager:
    """Manages Google OAuth2 credentials with automatic refresh"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def get_credentials(self, scopes: List[str], token_path: Optional[str] = None) -> Any:
        """
        Get valid Google credentials with automatic token refresh
        
        Args:
            scopes: List of OAuth2 scopes required
            token_path: Custom path for token storage
            
        Returns:
            Valid Google credentials object
        """
        if token_path is None:
            hash_digest = hashlib.md5(" ".join(sorted(scopes)).encode()).hexdigest()
            token_path = f"token_{hash_digest}.pickle"

        creds = None
        
        # Load existing credentials
        if os.path.exists(token_path):
            try:
                with open(token_path, "rb") as token_file:
                    creds = pickle.load(token_file)
                logger.info("Loaded existing credentials")
            except Exception as e:
                logger.warning(f"Failed to load existing credentials: {e}")

        # Validate and refresh credentials
        if not creds or not creds.valid or not set(scopes).issubset(set(creds.scopes or [])):
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    logger.info("Refreshed expired credentials")
                except Exception as e:
                    logger.warning(f"Failed to refresh credentials: {e}")
                    creds = None
            
            # Get new credentials if needed
            if not creds:
                logger.info("Acquiring new credentials")
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.config.credentials_path, scopes
                )
                creds = flow.run_local_server(port=0)
            
            # Save credentials
            try:
                with open(token_path, "wb") as token_file:
                    pickle.dump(creds, token_file)
                logger.info("Saved credentials")
            except Exception as e:
                logger.error(f"Failed to save credentials: {e}")

        return creds

class OpenAIManager:
    """Manages OpenAI API interactions"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def get_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "gpt-4", 
        temperature: float = 0.1,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> str:
        """
        Get completion from OpenAI API with error handling
        
        Args:
            messages: List of message dictionaries
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            stream: Whether to stream response
            
        Returns:
            Completion text
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if stream:
                content = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content += chunk.choices[0].delta.content
                        print(chunk.choices[0].delta.content, end="")
                print()  # New line after streaming
                return content
            else:
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"Error generating response: {e}"

class LifelogProcessor:
    """Processes lifelog data from Limitless API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def fetch_lifelogs(self, start_date: Optional[str] = None, end_date: Optional[str] = None, limit: int = 300) -> List[Dict[str, Any]]:
        """
        Fetch lifelogs for date range (defaults to 2 days ago through today)
        
        Args:
            start_date: Start date in YYYY-MM-DD format (defaults to 2 days ago)
            end_date: End date in YYYY-MM-DD format (defaults to today)
            limit: Maximum number of lifelogs to fetch per date
            
        Returns:
            List of lifelog dictionaries from all dates
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=2)).date().isoformat()
        if end_date is None:
            end_date = datetime.now().date().isoformat()
        
        all_lifelogs = []
        current_date = datetime.fromisoformat(start_date).date()
        end_date_obj = datetime.fromisoformat(end_date).date()
        
        while current_date <= end_date_obj:
            date_str = current_date.isoformat()
            try:
                logger.info(f"Fetching lifelogs for date: {date_str}")
                daily_lifelogs = get_lifelogs(
                    api_key=self.api_key,
                    date=date_str,
                    limit=limit
                )
                
                # Add date context to each lifelog
                for lifelog in daily_lifelogs:
                    lifelog['fetch_date'] = date_str
                
                all_lifelogs.extend(daily_lifelogs)
                logger.info(f"Fetched {len(daily_lifelogs)} lifelogs for {date_str}")
                
            except Exception as e:
                logger.error(f"Error fetching lifelogs for {date_str}: {e}")
            
            current_date += timedelta(days=1)
        
        logger.info(f"Successfully fetched {len(all_lifelogs)} total lifelogs across date range")
        return all_lifelogs
    
    def analyze_lifelogs(self, lifelogs: List[Dict[str, Any]], openai_manager: OpenAIManager) -> str:
        """
        Generate detailed analysis of lifelogs from multiple days with intelligent chunking
        
        Args:
            lifelogs: List of lifelog dictionaries
            openai_manager: OpenAI manager instance
            
        Returns:
            Detailed analysis text
        """
        if not lifelogs:
            return (
                "❌ NO CONVERSATION DATA AVAILABLE\n\n"
                "To extract commitments and promises, I need access to conversation transcripts. "
                "No lifelog data was provided for analysis. Please check:\n"
                "• Limitless API connection\n"
                "• API key validity\n"
                "• Date range settings\n"
                "• _client.py configuration"
            )

        # DEBUG: Let's see what's actually in the lifelog data
        print(f"\n🔍 Processing {len(lifelogs)} lifelog entries...")
        
        # Group lifelogs by date for better organization
        lifelogs_by_date = {}
        total_content_length = 0
        
        for log in lifelogs:
            fetch_date = log.get('fetch_date', 'unknown')
            if fetch_date not in lifelogs_by_date:
                lifelogs_by_date[fetch_date] = []
            lifelogs_by_date[fetch_date].append(log)

        # Extract and prepare conversation content with intelligent chunking
        all_conversations = []
        has_any_content = False
        
        for date in sorted(lifelogs_by_date.keys()):
            daily_logs = lifelogs_by_date[date]
            
            for i, log in enumerate(daily_logs):
                # Try multiple possible field names for transcript content
                transcript_content = None
                
                # Check the actual field that contains the data
                for field_name in ['contents', 'transcript', 'text', 'content', 'conversation', 'speech', 'audio_transcript', 'markdown']:
                    if field_name in log and log[field_name]:
                        transcript_content = str(log[field_name]).strip()
                        if transcript_content and transcript_content.lower() not in ['', 'none', 'null']:
                            break
                
                if transcript_content:
                    start_time = log.get('startTime', 'Unknown time')
                    title = log.get('title', f'Conversation {i+1}')
                    
                    conversation = {
                        'date': date,
                        'title': title,
                        'start_time': start_time,
                        'content': transcript_content,
                        'length': len(transcript_content)
                    }
                    
                    all_conversations.append(conversation)
                    total_content_length += len(transcript_content)
                    has_any_content = True
        
        if not has_any_content:
            return (
                "❌ INCOMPLETE CONVERSATION DATA\n\n"
                f"Found {len(lifelogs)} lifelog entries across {len(lifelogs_by_date)} days, "
                "but none contain conversation transcripts in expected fields.\n\n"
                "Checked fields: contents, transcript, text, content, conversation, speech, audio_transcript, markdown\n\n"
                "Please check _client.py get_lifelogs() function implementation."
            )

        print(f"✅ Found {len(all_conversations)} conversations with content")
        print(f"📊 Total content length: {total_content_length:,} characters")
        
        # If content is too large, we need to chunk it intelligently
        # Increased chunk size to preserve more context per conversation
        max_chars_per_chunk = 25000  # Increased from 15000 to preserve context
        
        if total_content_length > max_chars_per_chunk:
            print(f"⚠️ Content too large ({total_content_length:,} chars), chunking for analysis...")
            return self._analyze_conversations_in_chunks(all_conversations, openai_manager, max_chars_per_chunk, total_content_length)
        else:
            print("📝 Analyzing all conversations in single request...")
            return self._analyze_single_chunk(all_conversations, openai_manager)
    
    def _analyze_conversations_in_chunks(
        self, 
        conversations: List[Dict[str, Any]], 
        openai_manager: OpenAIManager, 
        max_chars_per_chunk: int,
        total_content_length: int
    ) -> str:
        """Analyze conversations in manageable chunks while preserving context"""
        
        # Sort conversations by date and length to keep related content together
        conversations.sort(key=lambda x: (x['date'], x['start_time']))
        
        chunks = []
        current_chunk = []
        current_chunk_size = 0
        
        for conv in conversations:
            # Calculate size including headers and formatting
            conv_header_size = len(f"--- {conv['title']} ({conv['date']}) ---\nTime: {conv['start_time']}\n")
            total_conv_size = conv['length'] + conv_header_size + 100  # Buffer for formatting
            
            # If this single conversation is too large, we need to handle it specially
            if total_conv_size > max_chars_per_chunk:
                # If we have a current chunk, process it first
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_chunk_size = 0
                
                # Split this large conversation into meaningful parts while preserving context
                large_conv_chunks = self._split_large_conversation(conv, max_chars_per_chunk)
                chunks.extend(large_conv_chunks)
                continue
            
            # Check if adding this conversation would exceed the limit
            if current_chunk_size + total_conv_size > max_chars_per_chunk and current_chunk:
                # Process current chunk
                chunks.append(current_chunk)
                current_chunk = [conv]
                current_chunk_size = total_conv_size
            else:
                # Add to current chunk
                current_chunk.append(conv)
                current_chunk_size += total_conv_size
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        print(f"📦 Split into {len(chunks)} chunks for analysis")
        print(f"📏 Average chunk size: {total_content_length // len(chunks):,} characters")
        
        # Analyze each chunk
        chunk_analyses = []
        for i, chunk in enumerate(chunks):
            chunk_size = sum(conv['length'] for conv in chunk if isinstance(chunk, list))
            if not isinstance(chunk, list):
                chunk_size = len(str(chunk))
            
            print(f"🔄 Analyzing chunk {i+1}/{len(chunks)} ({len(chunk) if isinstance(chunk, list) else 1} conversations, {chunk_size:,} chars)...")
            
            try:
                if isinstance(chunk, list):
                    analysis = self._analyze_single_chunk(chunk, openai_manager, chunk_number=i+1)
                else:
                    # Handle pre-split large conversation chunk
                    analysis = chunk
                chunk_analyses.append(analysis)
            except Exception as e:
                print(f"⚠️ Error analyzing chunk {i+1}: {e}")
                chunk_analyses.append(f"❌ Error analyzing chunk {i+1}: {e}")
        
        # Combine all chunk analyses
        print("🔄 Combining chunk analyses...")
        return self._combine_chunk_analyses(chunk_analyses, openai_manager)
    
    def _split_large_conversation(self, conv: Dict[str, Any], max_chars: int) -> List[str]:
        """Split a very large conversation while preserving context"""
        
        content = conv['content']
        title = conv['title']
        date = conv['date']
        time = conv['start_time']
        
        # Try to split on natural conversation boundaries
        # Look for speaker changes, paragraph breaks, or time stamps
        split_patterns = [
            '\n\n',  # Paragraph breaks
            '. ',    # Sentence endings
            '? ',    # Question endings
            '! ',    # Exclamation endings
        ]
        
        chunks = []
        current_pos = 0
        overlap_size = 500  # Characters to overlap between chunks for context
        
        while current_pos < len(content):
            # Calculate chunk end position
            chunk_end = min(current_pos + max_chars - 1000, len(content))  # Leave room for headers
            
            # Try to find a good breaking point
            best_break = chunk_end
            for pattern in split_patterns:
                # Look backwards from the end for a natural break
                for i in range(min(500, chunk_end - current_pos)):  # Search back up to 500 chars
                    check_pos = chunk_end - i
                    if check_pos > current_pos and content[check_pos-len(pattern):check_pos] == pattern:
                        best_break = check_pos
                        break
                if best_break < chunk_end:
                    break
            
            # Extract chunk content
            chunk_content = content[current_pos:best_break]
            
            # Add context and headers
            context_note = ""
            if current_pos > 0:
                context_note = f"\n[CONTINUING FROM PREVIOUS CHUNK]\n"
            if best_break < len(content):
                context_note += f"\n[CONVERSATION CONTINUES IN NEXT CHUNK]\n"
            
            chunk_text = f"--- {title} ({date}) - Part {len(chunks)+1} ---\nTime: {time}{context_note}\n{chunk_content}"
            chunks.append(chunk_text)
            
            # Move to next chunk with overlap for context
            if best_break >= len(content):
                break
            current_pos = max(best_break - overlap_size, best_break)
        
        print(f"  📄 Split large conversation '{title}' into {len(chunks)} parts")
        return chunks
    
    def _analyze_single_chunk(
        self, 
        conversations: List[Dict[str, Any]], 
        openai_manager: OpenAIManager,
        chunk_number: Optional[int] = None
    ) -> str:
        """Analyze a single chunk of conversations"""
        
        # Build the conversation text
        conversation_text = []
        date_range_start = min(conv['date'] for conv in conversations)
        date_range_end = max(conv['date'] for conv in conversations)
        
        for conv in conversations:
            header = f"--- {conv['title']} ({conv['date']}) ---"
            header += f"\nTime: {conv['start_time']}"
            header += f"\n{conv['content']}"
            conversation_text.append(header)
        
        chunk_suffix = f" (Chunk {chunk_number})" if chunk_number else ""
        date_range = f"{date_range_start} to {date_range_end}" if date_range_start != date_range_end else date_range_start
        
        user_prompt = f"Analyze Jarrett's conversations from {date_range}{chunk_suffix}:\n\n" + "\n\n".join(conversation_text)
        
        messages = [
            {"role": "system", "content": PromptTemplates.DETAILED_LIFELOG_SUMMARY},
            {"role": "user", "content": user_prompt}
        ]

        return openai_manager.get_completion(
            messages=messages,
            model="gpt-4o-mini",  # Use more efficient model for large content
            max_tokens=2000,
            temperature=0.1
        )
    
    def _combine_chunk_analyses(self, chunk_analyses: List[str], openai_manager: OpenAIManager) -> str:
        """Combine multiple chunk analyses into a comprehensive summary"""
        
        if len(chunk_analyses) == 1:
            return chunk_analyses[0]
        
        combination_prompt = f"""You are Jarrett's executive assistant. I have analyzed his conversations in {len(chunk_analyses)} separate chunks. 
Please combine these analyses into a single comprehensive summary using the same format.

CHUNK ANALYSES:
""" + "\n\n" + "="*50 + "\n\n".join([f"CHUNK {i+1}:\n{analysis}" for i, analysis in enumerate(chunk_analyses)])

        combination_prompt += """

INSTRUCTIONS:
1. Merge all commitments and promises (remove duplicates)
2. Combine all key wins
3. Provide one overall improvement suggestion
4. Note any contradictions across all conversations
5. Create a unified high-priority action list
6. Use the same format as individual analyses"""

        messages = [
            {
                "role": "system", 
                "content": "You are an expert at combining and synthesizing multiple conversation analyses into a comprehensive summary."
            },
            {"role": "user", "content": combination_prompt}
        ]

        return openai_manager.get_completion(
            messages=messages,
            model="gpt-4o-mini",
            max_tokens=3000,
            temperature=0.1
        )

class CalendarManager:
    """Manages Google Calendar operations"""
    
    def __init__(self, credentials):
        self.service = build('calendar', 'v3', credentials=credentials)
    
    def fetch_events(self, calendar_ids: List[str], target_date: datetime.date) -> List[Dict[str, Any]]:
        """
        Fetch calendar events for specified date
        
        Args:
            calendar_ids: List of calendar IDs to fetch from
            target_date: Date to fetch events for
            
        Returns:
            List of calendar event dictionaries
        """
        start_of_day = datetime.combine(target_date, datetime.min.time())
        start_of_day_cst = CST.localize(start_of_day)
        end_of_day_cst = start_of_day_cst + timedelta(days=1)

        start_iso = start_of_day_cst.astimezone(timezone.utc).isoformat()
        end_iso = end_of_day_cst.astimezone(timezone.utc).isoformat()

        all_events = []
        
        for cal_id in calendar_ids:
            try:
                logger.info(f"Fetching events from calendar: {cal_id}")
                events_result = self.service.events().list(
                    calendarId=cal_id,
                    timeMin=start_iso,
                    timeMax=end_iso,
                    singleEvents=True,
                    orderBy='startTime'
                ).execute()
                
                events = events_result.get('items', [])
                filtered_events = self._filter_events(events)
                all_events.extend(filtered_events)
                
                logger.info(f"Found {len(filtered_events)} valid events from {cal_id}")
                
            except Exception as e:
                logger.error(f"Error fetching calendar {cal_id}: {e}")

        # Sort by start time
        all_events.sort(key=lambda e: e.get('start', {}).get('dateTime', ''))
        logger.info(f"Total calendar events: {len(all_events)}")
        return all_events
    
    def _filter_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out cancelled and declined events"""
        filtered = []
        
        for event in events:
            # Skip cancelled events
            if event.get('status') == 'cancelled':
                continue
            
            # Check if user declined the event
            attendees = event.get('attendees', [])
            user_declined = False
            
            if attendees:
                # This would need to be enhanced to get actual user email
                # For now, we'll assume events without declined status are valid
                pass
            
            filtered.append(event)
        
        return filtered

class EmailManager:
    """Manages Gmail operations"""
    
    def __init__(self, credentials):
        self.service = build('gmail', 'v1', credentials=credentials)
    
    def fetch_emails(self) -> Dict[str, List[str]]:
        """
        Fetch emails from various categories
        
        Returns:
            Dictionary with email categories and their content
        """
        yesterday = (datetime.now(CST) - timedelta(days=1)).strftime('%Y/%m/%d')
        today = datetime.now(CST).strftime('%Y/%m/%d')
        tomorrow = (datetime.now(CST) + timedelta(days=1)).strftime('%Y/%m/%d')

        queries = {
            "inbox": "in:inbox",
            "sent": f"label:sent after:{yesterday} before:{tomorrow}",
            "priority": "label:\"! - Jarrett\" is:unread OR label:\"! - JD\" is:unread"
        }

        emails = {}

        for category, query in queries.items():
            try:
                logger.info(f"Fetching {category} emails...")
                max_results = 50 if category == "inbox" else 20
                
                response = self.service.users().messages().list(
                    userId='me', 
                    q=query, 
                    maxResults=max_results
                ).execute()
                
                messages = response.get('messages', [])
                extracted = self._extract_email_content(messages)
                emails[category] = extracted
                
                logger.info(f"Found {len(extracted)} {category} emails")
                
            except Exception as e:
                logger.error(f"Error fetching {category} emails: {e}")
                emails[category] = []

        return emails
    
    def _extract_email_content(self, messages: List[Dict[str, str]]) -> List[str]:
        """Extract content from email messages"""
        extracted = []
        
        for msg in messages:
            try:
                msg_data = self.service.users().messages().get(
                    userId='me', 
                    id=msg['id'], 
                    format='full'
                ).execute()
                
                headers = msg_data['payload'].get('headers', [])
                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
                sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
                snippet = msg_data.get('snippet', '')
                date_header = next((h['value'] for h in headers if h['name'] == 'Date'), '')

                email_content = f"Subject: {subject}\nFrom: {sender}\nDate: {date_header}\nSnippet: {snippet}"
                extracted.append(email_content)
                
            except Exception as e:
                logger.error(f"Error processing email {msg['id']}: {e}")

        return extracted

class TaskManager:
    """Manages Google Sheets task operations"""
    
    def __init__(self, credentials, spreadsheet_id: str):
        self.service = build('sheets', 'v4', credentials=credentials)
        self.spreadsheet_id = spreadsheet_id
    
    def fetch_active_tasks(self, sheet_name: str = "Todo Tasks") -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch active tasks from Google Sheets
        
        Args:
            sheet_name: Name of the sheet containing tasks
            
        Returns:
            Dictionary of tasks organized by area
        """
        try:
            logger.info(f"Fetching tasks from sheet: {sheet_name}")
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id, 
                range=sheet_name
            ).execute()
            
            values = result.get('values', [])
            if not values or len(values) < 2:
                logger.warning("No task data found in sheet")
                return {}

            return self._parse_task_data(values)
            
        except Exception as e:
            logger.error(f"Error fetching tasks: {e}")
            return {}
    
    def _parse_task_data(self, values: List[List[str]]) -> Dict[str, List[Dict[str, Any]]]:
        """Parse task data from sheet values"""
        headers = values[0]
        rows = values[1:]
        
        # Find column indices
        column_indices = {}
        for i, header in enumerate(headers):
            column_indices[header.lower()] = i
        
        tasks_by_area = {}
        now = datetime.now()
        next_week = now + timedelta(days=7)

        for row in rows:
            if not row:
                continue
                
            # Extract task data safely
            task_data = {}
            for col_name, col_index in column_indices.items():
                if col_index < len(row):
                    task_data[col_name] = row[col_index].strip()
                else:
                    task_data[col_name] = ""
            
            # Filter by due date
            due_str = task_data.get('due', '').strip()
            if due_str:
                try:
                    due_date = parse_date(due_str).date()
                    if due_date > next_week.date():
                        continue  # Skip tasks due beyond next week
                except Exception:
                    continue  # Skip tasks with invalid dates
            
            # Organize by area
            area = task_data.get('area', 'Uncategorized')
            if area not in tasks_by_area:
                tasks_by_area[area] = []
            
            tasks_by_area[area].append(task_data)

        total_tasks = sum(len(tasks) for tasks in tasks_by_area.values())
        logger.info(f"Loaded {total_tasks} active tasks")
        return tasks_by_area
    
    def add_task(self, sheet_name: str, task_data: List[str]) -> bool:
        """
        Add a new task to the sheet
        
        Args:
            sheet_name: Name of the sheet to add to
            task_data: List of values to add as a new row
            
        Returns:
            True if successful, False otherwise
        """
        try:
            body = {'values': [task_data]}
            result = self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range=sheet_name,
                valueInputOption='USER_ENTERED',
                insertDataOption='INSERT_ROWS',
                body=body
            ).execute()
            
            logger.info(f"Task added successfully: {task_data[0][:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error adding task: {e}")
            return False

class TaskExtractor:
    """Extracts new tasks from various data sources"""
    
    def __init__(self, openai_manager: OpenAIManager, task_manager: TaskManager, config: Config):
        self.openai_manager = openai_manager
        self.task_manager = task_manager
        self.config = config
    
    def extract_and_add_tasks(
        self, 
        lifelogs: List[Dict[str, Any]], 
        emails: Dict[str, List[str]], 
        calendar_events: List[Dict[str, Any]], 
        existing_tasks: Dict[str, List[Dict[str, Any]]], 
        lifelog_analysis: str
    ) -> int:
        """
        Extract new tasks from data sources and add to sheet
        
        Returns:
            Number of tasks added
        """
        logger.info("Starting task extraction process")
        
        # Prepare data for AI analysis
        context_data = self._prepare_context_data(
            lifelogs, emails, calendar_events, existing_tasks, lifelog_analysis
        )
        
        # Get AI task extraction
        extracted_tasks = self._extract_tasks_with_ai(context_data, existing_tasks)
        
        if not extracted_tasks:
            logger.info("No new tasks extracted")
            return 0
        
        # Get user confirmation
        if not self._confirm_task_addition(extracted_tasks):
            logger.info("Task addition cancelled by user")
            return 0
        
        # Add tasks to sheet
        return self._add_tasks_to_sheet(extracted_tasks)
    
    def _prepare_context_data(
        self, 
        lifelogs: List[Dict[str, Any]], 
        emails: Dict[str, List[str]], 
        calendar_events: List[Dict[str, Any]], 
        existing_tasks: Dict[str, List[Dict[str, Any]]], 
        lifelog_analysis: str
    ) -> str:
        """Prepare context data for AI analysis"""
        
        calendar_data = "\n".join([
            f"• {event.get('summary', 'No Title')}" 
            for event in calendar_events[:5]
        ])
        
        email_data = ""
        for category, email_list in emails.items():
            if email_list:
                email_data += f"\n=== {category.upper()} ===\n"
                for email in email_list[:3]:
                    email_data += email[:150] + "...\n"
        
        existing_sample = []
        for area, tasks in existing_tasks.items():
            for task in tasks[:3]:
                existing_sample.append(task.get('task', task.get('Task', 'Unknown task')))
        
        context = f"""Date: {datetime.now().strftime('%Y-%m-%d')}

CALENDAR EVENTS:
{calendar_data}

EMAILS:
{email_data}

CONVERSATION ANALYSIS:
{lifelog_analysis[:1000]}...

EXISTING TASKS (to avoid duplicating):
{chr(10).join([f"- {task}" for task in existing_sample[:10]])}

Extract specific commitments Jarrett made to named individuals."""

        return context
    
    def _extract_tasks_with_ai(
        self, 
        context_data: str, 
        existing_tasks: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Use AI to extract tasks from context data"""
        
        messages = [
            {"role": "system", "content": PromptTemplates.TASK_EXTRACTION},
            {"role": "user", "content": context_data}
        ]
        
        try:
            response = self.openai_manager.get_completion(
                messages=messages,
                model="gpt-4",
                temperature=0.1,
                max_tokens=800
            )
            
            # Extract JSON from response
            json_pattern = r'\[[\s\S]*?\]'
            match = re.search(json_pattern, response)
            
            if not match:
                logger.warning("No valid JSON found in AI response")
                return []
            
            tasks = json.loads(match.group(0))
            if not isinstance(tasks, list):
                logger.warning("AI response is not a list")
                return []
            
            # Filter duplicates
            filtered_tasks = self._filter_duplicate_tasks(tasks, existing_tasks)
            
            logger.info(f"Extracted {len(filtered_tasks)} unique tasks from {len(tasks)} total")
            return filtered_tasks
            
        except Exception as e:
            logger.error(f"Error in AI task extraction: {e}")
            return []
    
    def _filter_duplicate_tasks(
        self, 
        new_tasks: List[Dict[str, Any]], 
        existing_tasks: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Filter out duplicate tasks"""
        
        # Collect all existing task texts
        existing_task_texts = []
        for area, tasks in existing_tasks.items():
            for task in tasks:
                task_text = task.get('task', task.get('Task', '')).lower().strip()
                existing_task_texts.append(task_text)
        
        filtered_tasks = []
        
        for task in new_tasks:
            if not isinstance(task, dict) or 'task' not in task:
                continue
            
            task_text = task['task'].lower().strip()
            is_duplicate = False
            
            # Check for duplicates using similarity threshold
            for existing_text in existing_task_texts:
                if self._calculate_similarity(task_text, existing_text) > self.config.task_similarity_threshold:
                    logger.debug(f"Filtered duplicate task: {task['task'][:50]}...")
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_tasks.append(task)
        
        return filtered_tasks
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _confirm_task_addition(self, tasks: List[Dict[str, Any]]) -> bool:
        """Get user confirmation for task addition"""
        
        print("\n" + "="*60)
        print("📋 NEW TASKS READY TO ADD:")
        print("="*60)
        
        for i, task in enumerate(tasks, 1):
            print(f"\n{i}. {task.get('task', 'No description')}")
            print(f"   Priority: {task.get('priority', 'Not set')}")
            print(f"   Area: {task.get('area', 'Not set')}")
            print(f"   Due: {task.get('due_date', 'Today')}")
            print(f"   Notes: {task.get('notes', 'None')}")
        
        while True:
            response = input(f"\nAdd these {len(tasks)} tasks to Google Sheet? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no")
    
    def _add_tasks_to_sheet(self, tasks: List[Dict[str, Any]]) -> int:
        """Add tasks to Google Sheet"""
        
        today_str = datetime.now().strftime('%Y-%m-%d')
        tasks_added = 0
        
        for i, task in enumerate(tasks):
            logger.info(f"Adding task {i+1}/{len(tasks)}")
            
            # Prepare task data for sheet
            task_data = [
                task.get('task', ''),
                task.get('priority', ''),
                'Jarrett Davidson',
                task.get('area', ''),
                'Not started',
                '',  # Deliverable
                task.get('notes', ''),
                today_str,  # Date
                task.get('due_date', today_str),  # Due
                '',  # Email Reminder
                '',  # Frequency
                ''   # Last Email Sent
            ]
            
            if self.task_manager.add_task("Todo Tasks", task_data):
                tasks_added += 1
            
        logger.info(f"Successfully added {tasks_added}/{len(tasks)} tasks")
        return tasks_added

class SummaryGenerator:
    """Generates comprehensive leadership intelligence summaries"""
    
    def __init__(self, openai_manager: OpenAIManager, config: Config):
        self.openai_manager = openai_manager
        self.config = config
    
    def generate_summary(
        self,
        lifelogs: List[Dict[str, Any]],
        emails: Dict[str, List[str]],
        calendar_events: List[Dict[str, Any]],
        tasks: Dict[str, List[Dict[str, Any]]],
        lifelog_analysis: str
    ) -> str:
        """
        Generate comprehensive leadership intelligence summary
        
        Args:
            lifelogs: Lifelog data
            emails: Email data by category
            calendar_events: Calendar events
            tasks: Tasks organized by area
            lifelog_analysis: Detailed lifelog analysis
            
        Returns:
            Formatted summary text
        """
        logger.info("Generating leadership intelligence summary")
        
        # Prepare summary data
        summary_data = self._prepare_summary_data(
            lifelogs, emails, calendar_events, tasks, lifelog_analysis
        )
        
        # Generate AI summary (without full lifelog analysis to avoid duplication)
        ai_summary = self._generate_ai_summary(summary_data)
        
        # Create comprehensive output WITHOUT detailed conversation analysis
        comprehensive_summary = self._create_comprehensive_output(
            ai_summary, summary_data, ""  # Pass empty string instead of lifelog_analysis
        )
        
        # Save and open summary
        self._save_and_open_summary(comprehensive_summary)
        
        return comprehensive_summary
    
    def _prepare_summary_data(
        self,
        lifelogs: List[Dict[str, Any]],
        emails: Dict[str, List[str]],
        calendar_events: List[Dict[str, Any]],
        tasks: Dict[str, List[Dict[str, Any]]],
        lifelog_analysis: str
    ) -> Dict[str, Any]:
        """Prepare data for summary generation"""
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H:%M")
        
        # Calendar summary
        calendar_summary = ""
        for event in calendar_events:
            summary = event.get('summary', 'No Title')
            start_time = event.get('start', {}).get('dateTime', None)
            
            if start_time:
                try:
                    dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    time_formatted = dt.strftime("%I:%M %p")
                except:
                    time_formatted = "Time Unknown"
            else:
                time_formatted = "Time Unknown"
            
            attendees = event.get('attendees', [])
            if attendees:
                attendee_emails = ", ".join([a.get('email', '') for a in attendees[:3]])
                calendar_summary += f"• {time_formatted} - {summary} (Attendees: {attendee_emails})\n"
            else:
                calendar_summary += f"• {time_formatted} - {summary}\n"
        
        if not calendar_summary:
            calendar_summary = "• No events scheduled\n"
        
        # Email summary
        email_summary = ""
        for category, email_list in emails.items():
            if email_list:
                email_summary += f"\n=== {category.upper()} EMAILS ===\n"
                limit = 3 if category == 'priority' else 2
                for email in email_list[:limit]:
                    truncated = email[:150] + "..." if len(email) > 150 else email
                    email_summary += truncated + "\n"
        
        # Task summary
        priority_tasks = []
        for area, task_list in tasks.items():
            for task in task_list:
                priority = task.get('priority', task.get('Priority', ''))
                if priority in ['P1', 'P2']:
                    task_text = task.get('task', task.get('Task', 'No description'))
                    due_date = task.get('due', task.get('Due', ''))
                    
                    task_line = f"• {task_text}"
                    if priority:
                        task_line += f" ({priority}"
                        if due_date:
                            task_line += f", Due: {due_date}"
                        task_line += ")"
                    
                    priority_tasks.append(task_line)
        
        task_summary = "\n".join(priority_tasks[:8]) if priority_tasks else "• No high priority tasks"
        
        return {
            'date_str': date_str,
            'time_str': time_str,
            'total_conversations': len(lifelogs),
            'total_emails': sum(len(email_list) for email_list in emails.values()),
            'total_events': len(calendar_events),
            'calendar_summary': calendar_summary,
            'email_summary': email_summary,
            'task_summary': task_summary,
            'lifelog_analysis': lifelog_analysis
        }
    
    def _generate_ai_summary(self, summary_data: Dict[str, Any]) -> str:
        """Generate AI-powered summary"""
        
        system_prompt = PromptTemplates.LEADERSHIP_SUMMARY.format(
            date_str=summary_data['date_str'],
            time_str=summary_data['time_str'],
            total_conversations=summary_data['total_conversations'],
            total_emails=summary_data['total_emails'],
            total_events=summary_data['total_events']
        )
        
        user_prompt = f"""Generate Jarrett's leadership intelligence summary using this data:

=== TODAY'S CALENDAR ===
{summary_data['calendar_summary']}

=== EMAILS ===
{summary_data['email_summary']}

=== HIGH PRIORITY TASKS FROM TODO SYSTEM ===
{summary_data['task_summary']}

=== CONVERSATION SUMMARY FOR COMMITMENTS ===
Extract commitments from conversation analysis, but focus only on actionable items that should be scheduled or added to tasks.

INSTRUCTIONS:
1. Split conversation commitments into two categories:
   - CALENDAR items: mentions specific times/dates ('on Friday', 'tomorrow at 2pm', 'next week', 'by Monday')
   - TODO items: general commitments ('I will', 'I should', 'I need to', 'I'll follow up')
2. Avoid duplicating items between the calendar and todo sections
3. Focus on actionable intelligence with specific names and contexts
4. Extract people to follow up with from all sources (emails, conversations, tasks)
5. DO NOT include the full detailed conversation analysis - that goes in a separate file
6. Keep summary concise and actionable"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            return self.openai_manager.get_completion(
                messages=messages,
                model="gpt-4",
                max_tokens=3000
            )
        except Exception as e:
            logger.error(f"Error generating AI summary: {e}")
            return self._create_fallback_summary(summary_data)
    
    def _create_fallback_summary(self, summary_data: Dict[str, Any]) -> str:
        """Create fallback summary when AI is unavailable"""
        
        return f"""🎯 JARRETT'S LEADERSHIP INTELLIGENCE SUMMARY
================================================================================
📅 Analysis Date: {summary_data['date_str']}
🕐 Generated: {summary_data['time_str']}
📊 Data Sources: {summary_data['total_conversations']} conversations + {summary_data['total_emails']} emails + {summary_data['total_events']} events

📅 TODAY'S CALENDAR
----------------------------------------
{summary_data['calendar_summary'].strip()}

📋 HIGH-PRIORITY TASKS (from ToDo Task System)
----------------------------------------
{summary_data['task_summary']}

📧 EMAIL ACTION ITEMS
----------------------------------------
• AI analysis temporarily unavailable - refer to email data

📅 TO BE SCHEDULED IN CALENDAR
----------------------------------------
• AI analysis temporarily unavailable

📝 TO BE ADDED TO TODO TASKS
----------------------------------------
• AI analysis temporarily unavailable

👥 PEOPLE TO FOLLOW UP WITH
----------------------------------------
• AI analysis temporarily unavailable

🧠 LEADERSHIP INSIGHT
----------------------------------------
• AI analysis temporarily unavailable

📝 MEETING PREPARATION
----------------------------------------
• AI analysis temporarily unavailable

================================================================================
✅ Summary Complete - AI analysis temporarily unavailable"""
    
    def _create_comprehensive_output(
        self, 
        ai_summary: str, 
        summary_data: Dict[str, Any], 
        lifelog_analysis: str
    ) -> str:
        """Create streamlined output focused on actionable intelligence"""
        
        return f"""{ai_summary}

================================================================================
📊 DATA PROCESSING SUMMARY
================================================================================
• Lifelogs processed: {summary_data['total_conversations']}
• Emails analyzed: {summary_data['total_emails']}
• Calendar events: {summary_data['total_events']}
• Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

NOTE: Detailed conversation analysis with full commitments and promises 
saved separately as detailed lifelog analysis file.
================================================================================"""
    
    def _save_and_open_summary(self, summary_text: str) -> None:
        """Save summary to file and open it"""
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"leadership_summary_{timestamp}.txt"
        output_path = os.path.join(self.config.output_dir, filename)
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(summary_text)
            
            logger.info(f"Summary saved to: {output_path}")
            
            # Try to open the file
            import subprocess
            import platform
            
            if platform.system() == 'Windows':
                os.startfile(output_path)
            elif platform.system() == 'Darwin':
                subprocess.run(['open', output_path])
            else:
                subprocess.run(['xdg-open', output_path])
            
            logger.info(f"Automatically opened: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving/opening summary: {e}")

class LeadershipIntelligenceSystem:
    """Main orchestrator for the Leadership Intelligence System"""
    
    def __init__(self):
        self.config = Config()
        self.credentials_manager = GoogleCredentialsManager(self.config)
        self.openai_manager = OpenAIManager()
        
        # Initialize Google services
        self.credentials = None
        self.lifelog_processor = None
        self.calendar_manager = None
        self.email_manager = None
        self.task_manager = None
        self.task_extractor = None
        self.summary_generator = None
    
    def initialize(self) -> None:
        """Initialize all system components"""
        
        logger.info("Initializing Leadership Intelligence System")
        
        # Get Google credentials
        self.credentials = self.credentials_manager.get_credentials(SCOPES)
        
        # Initialize components
        limitless_api_key = os.getenv("LIMITLESS_API_KEY")
        self.lifelog_processor = LifelogProcessor(limitless_api_key)
        self.calendar_manager = CalendarManager(self.credentials)
        self.email_manager = EmailManager(self.credentials)
        self.task_manager = TaskManager(self.credentials, self.config.spreadsheet_id)
        self.task_extractor = TaskExtractor(
            self.openai_manager, self.task_manager, self.config
        )
        self.summary_generator = SummaryGenerator(self.openai_manager, self.config)
        
        logger.info("System initialization complete")
    
    def run_analysis(self) -> None:
        """Run the complete leadership intelligence analysis"""
        
        try:
            logger.info("Starting leadership intelligence analysis")
            
            # Step 1: Fetch and analyze lifelogs
            logger.info("Step 1: Fetching and analyzing lifelogs")
            lifelogs = self.lifelog_processor.fetch_lifelogs()
            lifelog_analysis = self.lifelog_processor.analyze_lifelogs(
                lifelogs, self.openai_manager
            )
            
            # Save detailed lifelog analysis as separate file
            if lifelog_analysis:
                save_detailed_lifelog_summary(lifelog_analysis)
                logger.info("Detailed lifelog analysis saved separately")
            
            # Step 2: Fetch calendar events
            logger.info("Step 2: Fetching calendar events")
            today_date = datetime.now(CST).date()
            calendar_events = self.calendar_manager.fetch_events(
                self.config.calendar_ids, today_date
            )
            
            # Step 3: Fetch emails
            logger.info("Step 3: Fetching emails")
            emails = self.email_manager.fetch_emails()
            
            # Step 4: Fetch existing tasks
            logger.info("Step 4: Fetching existing tasks")
            existing_tasks = self.task_manager.fetch_active_tasks()
            
            # Step 5: Extract and add new tasks
            logger.info("Step 5: Extracting new tasks")
            new_tasks_added = self.task_extractor.extract_and_add_tasks(
                lifelogs, emails, calendar_events, existing_tasks, lifelog_analysis
            )
            
            # Refresh tasks if new ones were added
            if new_tasks_added > 0:
                logger.info(f"Refreshing task list - added {new_tasks_added} new tasks")
                existing_tasks = self.task_manager.fetch_active_tasks()
            
            # Step 6: Generate comprehensive summary
            logger.info("Step 6: Generating leadership intelligence summary")
            summary = self.summary_generator.generate_summary(
                lifelogs, emails, calendar_events, existing_tasks, lifelog_analysis
            )
            
            # Print completion summary
            self._print_completion_summary(
                len(lifelogs), len(emails.get('inbox', [])) + len(emails.get('priority', [])),
                len(calendar_events), sum(len(tasks) for tasks in existing_tasks.values()),
                new_tasks_added
            )
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            raise
    
    def _print_completion_summary(
        self, 
        lifelog_count: int, 
        email_count: int, 
        event_count: int, 
        task_count: int, 
        new_tasks: int
    ) -> None:
        """Print analysis completion summary"""
        
        print("\n" + "="*80)
        print("🎯 LEADERSHIP INTELLIGENCE ANALYSIS COMPLETE")
        print("="*80)
        print(f"📊 Data Processed:")
        print(f"   • Lifelogs: {lifelog_count}")
        print(f"   • Emails: {email_count}")
        print(f"   • Calendar Events: {event_count}")
        print(f"   • Active Tasks: {task_count}")
        if new_tasks > 0:
            print(f"   • New Tasks Added: {new_tasks}")
        print("\n📁 Files Generated:")
        print("   • Detailed Lifelog Analysis")
        print("   • Comprehensive Leadership Intelligence Summary")
        print("\n📂 Files saved to Google Drive and automatically opened")
        print("="*80)

def save_detailed_lifelog_summary(summary_text: str) -> Optional[str]:
    """
    Save detailed lifelog summary to file
    
    Args:
        summary_text: The summary text to save
        
    Returns:
        Path to saved file or None if failed
    """
    try:
        output_dir = r"G:\My Drive\Jarrett Folders\JD INC\Limitle3ss\enhanced\Lifelogs output"
        os.makedirs(output_dir, exist_ok=True)
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H%M")
        filename = f"{date_str}_{time_str}_detailed_lifelog_analysis.txt"
        output_path = os.path.join(output_dir, filename)
        
        print(f"💾 Saving detailed lifelog analysis to: {output_path}")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        
        logger.info(f"Detailed lifelog analysis saved to: {output_path}")
        
        # Try to open the file
        import subprocess
        import platform
        
        try:
            if platform.system() == 'Windows':
                os.startfile(output_path)
            elif platform.system() == 'Darwin':
                subprocess.run(['open', output_path])
            else:
                subprocess.run(['xdg-open', output_path])
            
            print(f"📂 Automatically opened detailed analysis file")
        except Exception as e:
            print(f"⚠️ Could not auto-open file: {e}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving detailed lifelog summary: {e}")
        print(f"❌ Error saving detailed lifelog analysis: {e}")
        return None

def main():
    """Main entry point for the Leadership Intelligence System"""
    
    print("🚀 JARRETT'S ENHANCED LEADERSHIP INTELLIGENCE SYSTEM")
    print("="*80)
    
    try:
        # Create and initialize system
        system = LeadershipIntelligenceSystem()
        system.initialize()
        
        # Run analysis
        system.run_analysis()
        
    except EnvironmentError as e:
        logger.error(f"Environment configuration error: {e}")
        print(f"❌ Configuration Error: {e}")
        print("💡 Please check your .env file and API keys")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected Error: {e}")
        print("💡 Check the log file for detailed error information")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()