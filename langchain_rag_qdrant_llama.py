# At the top of your script
import nest_asyncio
nest_asyncio.apply()
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
# Updated memory import
from langchain.memory import ConversationBufferMemory
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Updated Qdrant import
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
import imaplib
import email
import pandas as pd
import plotly.express as px
import streamlit as st
from email.header import decode_header
from datetime import datetime, timedelta
import pytz
import time
import html
import re
import requests
import json
import threading
from itertools import chain
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from bs4 import BeautifulSoup
import os
import uuid
import logging
from streamlit_autorefresh import st_autorefresh
import traceback
from streamlit import cache_data
import numpy as np
from typing import List, Dict, Any, Optional
import hashlib
import pickle
from pathlib import Path
import plotly.graph_objects as go
import sys

# More comprehensive fix for torch._classes
class CustomTorchModule:
    def __init__(self):
        self._path = ["dummy_path"]  # Make _path an attribute directly
    
    def __getattr__(self, name):
        if name == "__path__":
            return self
        return None
    
    # Add __iter__ to properly handle iteration over _path
    def __iter__(self):
        return iter(self._path)

# Apply the fix if torch is imported
try:
    if 'torch._classes' in sys.modules:
        custom_module = CustomTorchModule()
        sys.modules['torch._classes'] = custom_module
except Exception as e:
    print(f"Error applying torch._classes fix: {e}")


# Suppress LangChain deprecation warnings
import warnings
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="langchain.*")

# LangChain and LLM imports

# Notification and refresh settings
NOTIFICATION_CONFIG = {
    'high_priority_refresh_seconds': 30,
    'medium_priority_refresh_seconds': 60,
    'low_priority_refresh_seconds': 300
}

# Qdrant configuration
QDRANT_HOST = "localhost"  # Docker container hostname
QDRANT_PORT = 6333  # Default Qdrant port
QDRANT_COLLECTION_NAME = "emails"
VECTOR_DIMENSION = 384  # Dimension for sentence-transformers models

# Configure logging
logging.basicConfig(filename='enhanced_email_dashboard.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

# Set page config as the first Streamlit command
st.set_page_config(page_title="Enhanced Mail Alerts Dashboard with LLM",
                   layout="wide", initial_sidebar_state="expanded")

# Azure AD app details (should be stored in environment variables for production)

# Set timezone to IST
IST = pytz.timezone("Asia/Kolkata")

# Thread-local storage for IMAP connections
thread_local = threading.local()

# Enhanced database with 30-day retention
DB_FILE = "enhanced_unresolved_alerts.db"
VECTOR_DB_PATH = "./vector_db"
# Replace FAISS index path with Qdrant reference
QDRANT_STORAGE_PATH = "./qdrant_storage"
# Updated to use Q2_K quantized model
LLM_MODEL_PATH = "./models/llama-2-7b-chat.Q2_K.gguf"

# Configuration flags
ENABLE_VECTOR_DB = True  # Vector database is now enabled
FAST_MODE = True  # Enable fast mode for quicker analysis
CACHE_LLM_RESPONSES = True  # Cache LLM responses for faster repeated queries
MAX_SIMILAR_EMAILS = 3  # Reduce from 5 to 3 for faster processing

# List of sender email IDs to filter (normalized to lowercase)


SENDERS = [sender.lower() for sender in SENDERS]

# Define the list of folders to fetch emails from
EMAIL_FOLDERS = [
    "INBOX", "INBOX/VuSmartMaps_2023_Sep_OLD", "INBOX/freshping",
    "INBOX/QK", "INBOX/EURONET", "INBOX/All_Alerts",
    "INBOX/ORACLE_PROD_WEBLOGIC_ALERT"
]

# Enhanced Critical keywords with severity levels and proactive detection
CRITICAL_KEYWORDS = {
    'HIGH': [
        "shutdown", "crash", "down", "stopped", "failure", "error", "uptime", "anomaly", "UPI transactions impact",
        "device down", "heartbeat", "oomkilled", "crashloopbackoff", "stuck", "abed",
        "service unavailable", "connection refused", "Critical - BLR-VxRail-", "AUF Fincare transactions impacted",
        "The number of hogged threads is", "timeout expired", "fatal error"
    ],

    'MEDIUM': [
        "problem", "high memory", "high disk usage",
        "restart", "bounce", "diskspace", "heap memory availability is less",
        "the number of hogged threads", "failure transaction rate",
        "imagepullbackoff", "stop",
        "degraded performance", "high cpu", "memory leak", "slow response"
    ],

    'LOW': [
        "maintenance", "scheduled", "planned", "notice",
        "backup", "update", "patch", "routine", "normal"
    ]
}

# Disable TinyLlama support and provide a single unified LLMConfig
TINYLLAMA_AVAILABLE = True


class LLMConfig:
    """LLM configuration with TinyLlama support."""

    def __init__(self):
        self.model_path = LLM_MODEL_PATH
        self.temperature = 0.1
        self.max_tokens = 1024 if FAST_MODE else 2048
        self.top_p = 0.9
        self.verbose = False
        self.n_ctx = 2048
        self.n_threads = 8 if FAST_MODE else 4
        # TinyLlama model path
        self.tinyllama_model_path = "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        # Added for lightweight model support
        self.use_lightweight_model = False
        self.lightweight_model_path = "./models/llama-2-7b-chat.Q4_0.gguf"

    def get_llm(self, model_type="standard"):
        """Initialize and return LlamaCpp model"""
        try:
            # If TinyLlama is available and requested, use it
            if TINYLLAMA_AVAILABLE and os.path.exists(self.tinyllama_model_path):
                model_path = self.tinyllama_model_path
                st.info("🤖 Using TinyLlama model for analysis")
            # Try lightweight model if enabled
            elif self.use_lightweight_model and os.path.exists(self.lightweight_model_path):
                model_path = self.lightweight_model_path
                st.info("🚀 Using lightweight model for faster processing")
            elif os.path.exists(self.model_path):
                model_path = self.model_path
            else:
                st.error(f"LLM model not found at {self.model_path}")
                st.info("📋 To fix this issue:")
                st.info("1. Run: python3 setup_tinyllama.py")
                st.info(
                    "2. Or manually download from: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
                st.info(
                    "3. Place the model file in the 'models' directory")
                st.info("4. Restart the application")
                return None

            return LlamaCpp(
                model_path=model_path,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                verbose=self.verbose,
                n_ctx=self.n_ctx,  # Use the full context window
                n_gpu_layers=0,  # Set to higher number if you have GPU
                n_threads=8 if FAST_MODE else 4,
                n_batch=512,
                repeat_penalty=1.1,
                f16_kv=True  # Use 16-bit key/value cache for better performance
            )
        except Exception as e:
            st.error(f"Failed to load LLM model: {e}")
            st.info("📋 To fix this issue:")
            st.info("1. Ensure you have enough RAM (4GB+ recommended for TinyLlama)")
            st.info("2. Check if the model file is corrupted")
            return None
			

# Vector Database Management

# In the init_vector_db method of VectorDBManager class
class VectorDBManager:
    """Class to manage vector database operations"""
    
    def __init__(self):
        self.client = None
        self.vector_db = None
        self.embeddings = self._init_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        self.init_vector_db()
    
    def _init_embeddings(self):
        """Initialize the embeddings model"""
        try:
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as e:
            st.error(f"Failed to initialize embeddings: {e}")
            return None
            
    def init_vector_db(self):
        """Initialize Qdrant vector database"""
        if self.embeddings is None:
            st.warning(
                "Vector database initialization skipped - embeddings not available")
            return

        try:
            # Check if client exists and is working
            if self.client:
                try:
                    self.client.get_collections()
                    # Client is working, no need to recreate
                    return
                except:
                    # Client exists but is closed, create a new one
                    pass
                
            # Connect to Qdrant
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if QDRANT_COLLECTION_NAME not in collection_names:
                # Create new collection if it doesn't exist
                self.client.create_collection(
                    collection_name=QDRANT_COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=VECTOR_DIMENSION,
                        distance=Distance.COSINE
                    )
                )
                st.info(f"📝 Created new Qdrant collection: {QDRANT_COLLECTION_NAME}")
                
                # Initialize empty Qdrant vectorstore with updated import
                self.vector_db = Qdrant(
                    client=self.client,
                    collection_name=QDRANT_COLLECTION_NAME,
                    embeddings=self.embeddings
                )
            else:
                # Use existing collection with updated import
                self.vector_db = Qdrant(
                    client=self.client,
                    collection_name=QDRANT_COLLECTION_NAME,
                    embeddings=self.embeddings
                )
                st.success(f"✅ Connected to existing Qdrant collection: {QDRANT_COLLECTION_NAME}")

        except Exception as e:
            st.error(f"Failed to initialize Qdrant vector database: {e}")
            st.info("Make sure Qdrant is running in Docker with the correct port mappings")
            st.code("docker run -d -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant")
            self.vector_db = None


    def add_emails_to_vector_db(self, emails_data: List[Dict[str, Any]]):
        """Add email data to Qdrant vector database"""
        if self.embeddings is None or self.vector_db is None:
            st.warning(
                "Vector database not available - embeddings not initialized")
            return

        try:
            documents = []
            metadatas = []

            for email_data in emails_data:
                # Create a comprehensive document from email data
                doc_text = f"""
                Date: {email_data.get('date', '')}
                Sender: {email_data.get('sender', '')}
                Subject: {email_data.get('subject', '')}
                Folder: {email_data.get('folder', '')}
                Body: {email_data.get('body', '')}
                Keywords: {email_data.get('keywords', '')}
                Severity: {email_data.get('severity', '')}
                Application: {email_data.get('application', '')}
                """

                # Split text into chunks
                chunks = self.text_splitter.split_text(doc_text)

                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadatas.append({
                        'email_id': email_data.get('id', ''),
                        'date': email_data.get('date', ''),
                        'sender': email_data.get('sender', ''),
                        'subject': email_data.get('subject', ''),
                        'folder': email_data.get('folder', ''),
                        'severity': email_data.get('severity', ''),
                        'application': email_data.get('application', ''),
                        'chunk_id': i
                    })

            if documents:
                # Add documents to Qdrant
                self.vector_db.add_texts(
                    texts=documents,
                    metadatas=metadatas
                )
                
                st.success(
                    f"✅ Added {len(documents)} email chunks to Qdrant vector database")

        except Exception as e:
            st.error(f"Failed to add emails to Qdrant vector database: {e}")
            st.info("Check Qdrant logs for more information")

    def search_similar_emails(self, query: str, k: int = MAX_SIMILAR_EMAILS):
        """Search for similar emails using Qdrant vector similarity or keyword fallback"""
        if self.vector_db is None:
            st.warning(
                "Vector database not available - using keyword search fallback")
            return self.keyword_search_fallback(query, k)

        try:
            # Ensure client connection is valid before searching
            if self.client:
                try:
                    self.client.get_collections()
                except:
                    # Reconnect if client was closed
                    self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
                    self.vector_db = Qdrant(
                        client=self.client, 
                        collection_name=QDRANT_COLLECTION_NAME,
                        embeddings=self.embeddings
                    )
                
            # Use Qdrant similarity search
            docs_with_scores = self.vector_db.similarity_search_with_score(query, k=k)
            return docs_with_scores
        except Exception as e:
            st.error(f"Failed to search Qdrant vector database: {e}")
            st.info("Falling back to keyword search...")
            return self.keyword_search_fallback(query, k)

    def keyword_search_fallback(self, query: str, k: int = MAX_SIMILAR_EMAILS):
        """Fallback keyword search when vector database is not available"""
        try:
            # Extract keywords from query
            query_terms = query.lower().split()
            if not query_terms:
                return[]

            # Build SQL query with conditions for each term
            conditions = []
            params = []

            for term in query_terms:
                if len(term) > 2:  # Skip very short terms
                    conditions.append("(subject LIKE ? OR body LIKE ?)")
                    term_param = f"%{term}%"
                    params.extend([term_param, term_param])
            
            if not conditions:
                conditions = ["(1=1)"]

            where_clause = " OR ".join(conditions)
            sql_query = f'''
                SELECT id, date, sender, subject, body, folder, severity, keywords, application
                FROM emails
                WHERE {where_clause}
                ORDER BY date DESC
                LIMIT ?
            '''
            params.append(k)
            conn = sqlite3.connect(DB_FILE)
            results = pd.read_sql_query(sql_query, conn, params=params)
            conn.close()
            
            if results.empty:
                return[]

            # Convert to document format for compatibility
            documents = []
            for _, row in results.iterrows():
                doc_content = f"Date: {row['date']}\nSender: {row['sender']}\nSubject: {row['subject']}\nBody: {row['body'][:500]}..."
                doc_metadata = {
                    'email_id': row['id'],
                    'date': row['date'],
                    'sender': row['sender'],
                    'subject': row['subject'],
                    'folder': row['folder'],
                    'severity': row['severity'],
                    'application': row['application']
                }

                try:
                    from langchain_core.documents import Document
                except ImportError:
                    try:
                        from langchain.schema.document import Document
                    except ImportError:
                        from langchain.schema import Document
                        
                doc = Document(page_content=doc_content, metadata=doc_metadata)
                documents.append((doc, 0.8))  # Default similarity score

            return documents
    
        except Exception as e:
            logging.error(f"Keyword search fallback failed: {e}")
            st.error(f"Keyword search fallback failed: {e}")
            return []
# Enhanced Database with 30-day retention


def init_enhanced_db():
    """Initialize enhanced database with 30-day retention"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Enhanced emails table with more fields
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS emails (
            id TEXT PRIMARY KEY,
            date TEXT NOT NULL,
            sender TEXT NOT NULL,
            subject TEXT NOT NULL,
            body TEXT,
            folder TEXT NOT NULL,
            severity TEXT,
            keywords TEXT,
            application TEXT,
            hash TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Alerts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id TEXT PRIMARY KEY,
            email_id TEXT,
            subject TEXT NOT NULL,
            sender TEXT NOT NULL,
            body_snippet TEXT,
            severity TEXT NOT NULL,
            keywords TEXT,
            application TEXT,
            priority_score REAL,
            is_resolved INTEGER DEFAULT 0,
            resolved_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (email_id) REFERENCES emails (id)
        )
    ''')

    # LLM interactions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS llm_interactions (
            id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            context_emails TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Last fetch tracking table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS last_fetch (
            folder TEXT PRIMARY KEY,
            last_fetch_time TEXT
        )
    ''')

    # RCA documents table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rca_documents (
            id TEXT PRIMARY KEY,
            alert_id TEXT,
            filename TEXT,
            file_path TEXT,
            upload_date TEXT,
            analysis_text TEXT,
            FOREIGN KEY (alert_id) REFERENCES alerts (id)
        )
    ''')

    # Create indexes for better performance
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS idx_emails_date ON emails(date)')
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS idx_emails_sender ON emails(sender)')
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS idx_emails_severity ON emails(severity)')
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)')
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON alerts(is_resolved)')

    conn.commit()
    conn.close()


def cleanup_old_data():
    """Clean up data older than 30 days"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Calculate 30 days ago
        thirty_days_ago = (datetime.now(IST) - timedelta(days=30)
                           ).strftime('%Y-%m-%d %H:%M:%S')

        # Delete old emails
        cursor.execute('DELETE FROM emails WHERE date < ?', (thirty_days_ago,))
        deleted_emails = cursor.rowcount

        # Delete old alerts
        cursor.execute('DELETE FROM alerts WHERE created_at < ?',
                       (thirty_days_ago,))
        deleted_alerts = cursor.rowcount

        # Delete old LLM interactions
        cursor.execute(
            'DELETE FROM llm_interactions WHERE created_at < ?', (thirty_days_ago,))
        deleted_interactions = cursor.rowcount

        # Delete old RCA documents
        cursor.execute(
            'DELETE FROM rca_documents WHERE upload_date < ?', (thirty_days_ago,))
        deleted_rca_docs = cursor.rowcount

        conn.commit()
        conn.close()

        if deleted_emails > 0 or deleted_alerts > 0 or deleted_interactions > 0 or deleted_rca_docs > 0:
            st.info(
                f"Cleaned up old data: {deleted_emails} emails, {deleted_alerts} alerts, {deleted_interactions} interactions, {deleted_rca_docs} RCA documents")

    except Exception as e:
        st.error(f"Failed to cleanup old data: {e}")

# Helper functions for last fetch tracking


def get_last_fetch_time(folder):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT last_fetch_time FROM last_fetch WHERE folder = ?', (folder,))
    row = c.fetchone()
    conn.close()
    if row and row[0]:
        return row[0]
    return None


def update_last_fetch_time(folder, last_fetch_time):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('INSERT OR REPLACE INTO last_fetch (folder, last_fetch_time) VALUES (?, ?)',
              (folder, last_fetch_time))
    conn.commit()
    conn.close()

# IMAP connection management


def get_imap_connection(access_token, folder, retries=5, initial_backoff=2):
    """Enhanced IMAP connection with robust error handling and cleanup"""
    auth_string = f"user={EMAIL_ACCOUNT}\x01auth=Bearer {access_token}\x01\x01"

    for attempt in range(retries):
        mail = None
        try:
            # Create fresh connection for each attempt
            mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)

            # Set socket timeout for better error handling
            mail.sock.settimeout(30)

            # Authenticate with OAuth2
            mail.authenticate("XOAUTH2", lambda x: auth_string.encode("utf-8"))

            # Verify authentication status
            status, data = mail.status('INBOX', '(MESSAGES)')
            if status != 'OK':
                raise Exception(
                    f"Authentication verification failed: {status}")

            # Select the target folder
            status, data = mail.select(folder)
            if status != 'OK':
                raise Exception(
                    f"Failed to select folder {folder}: {status} - {data}")

            logging.info(
                f"Successfully connected to folder {folder} on attempt {attempt + 1}")
            return mail, None

        except Exception as e:
            error_str = str(e)
            logging.error(
                f"Attempt {attempt + 1}/{retries} failed for folder {folder}: {error_str}")

            # Clean up failed connection
            if mail:
                try:
                    mail.close()
                    mail.logout()
                except:
                    pass

            # Handle specific error types
            if "authenticated but not connected" in error_str.lower():
                # This specific error needs longer backoff
                backoff_time = initial_backoff * (3 ** attempt)
                logging.warning(
                    f"Authentication/connection issue, backing off for {backoff_time} seconds")
                time.sleep(backoff_time)
            elif "throttled" in error_str.lower() or "rate limit" in error_str.lower():
                # Rate limiting - exponential backoff
                backoff_time = initial_backoff * (2 ** attempt)
                logging.warning(
                    f"Rate limiting detected, backing off for {backoff_time} seconds")
                time.sleep(backoff_time)
            elif attempt < retries - 1:
                # Other errors - shorter backoff for retry
                backoff_time = initial_backoff + (attempt * 2)
                logging.info(f"Retrying connection in {backoff_time} seconds")
                time.sleep(backoff_time)

    return None, f"Failed to connect to folder {folder} after {retries} retries"


def parse_sender(sender_str):
    if not sender_str:
        return "Unknown Sender"
    sender_str = sender_str.replace('"', '').strip()
    match = re.match(r"(.*)<(.+@.+)>", sender_str)
    if match:
        return match.group(2).lower().strip()
    return sender_str.lower().strip()


def fetch_email_direct(mail, num, folder):
    """Fetch individual email using existing IMAP connection with improved time handling"""
    try:
        result, msg_data = mail.fetch(num, "(RFC822)")
        if result != "OK" or not msg_data or not isinstance(msg_data[0], tuple):
            return None

        raw_email = msg_data[0][1]
        if not isinstance(raw_email, bytes):
            return None

        msg = email.message_from_bytes(raw_email)
        raw_sender = msg["From"]

        # Decode subject
        subject, encoding = decode_header(msg["Subject"])[0]
        subject = subject.decode(
            encoding or "utf-8") if isinstance(subject, bytes) else subject

        sender = parse_sender(raw_sender)

        # Parse email date with better timezone handling
        email_date = email.utils.parsedate_tz(msg["Date"])
        if email_date:
            # Convert to datetime object
            email_datetime = datetime(*email_date[:6])

            # Apply timezone offset if available
            if email_date[9] is not None:
                offset = timedelta(seconds=email_date[9])
                email_datetime = email_datetime - offset

            # Handle timezone properly by first converting to UTC then to IST
            try:
                # If not timezone aware, assume UTC
                email_datetime_utc = pytz.utc.localize(email_datetime)
                email_datetime_ist = email_datetime_utc.astimezone(IST)
            except ValueError:
                # Already timezone aware
                email_datetime_ist = pytz.utc.normalize(
                    email_datetime).astimezone(IST)

            # Cap future dates to now if more than 5 minutes ahead
            now_ist = datetime.now(IST)
            if email_datetime_ist > now_ist + timedelta(minutes=5):
                logging.warning(
                    f"Email date {email_datetime_ist} is in the future. Capping to now.")
                email_datetime_ist = now_ist

            date_str = email_datetime_ist.strftime("%Y-%m-%d %H:%M:%S")
        else:
            logging.warning(
                f"Invalid date for email from {sender} with subject '{subject}'")
            # Use current time as fallback
            date_str = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

        # Extract email body
        mail_body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    mail_body = part.get_payload(
                        decode=True).decode("utf-8", errors="ignore")
                    break
                elif content_type == "text/html":
                    html_content = part.get_payload(
                        decode=True).decode("utf-8", errors="ignore")
                    soup = BeautifulSoup(html_content, 'html.parser')
                    mail_body = soup.get_text(separator=' ', strip=True)
                    break
        else:
            content_type = msg.get_content_type()
            payload = msg.get_payload(decode=True).decode(
                "utf-8", errors="ignore")
            if content_type == "text/plain":
                mail_body = payload
            elif content_type == "text/html":
                soup = BeautifulSoup(payload, 'html.parser')
                mail_body = soup.get_text(separator=' ', strip=True)

        return [date_str, sender, subject, mail_body.strip(), folder]

    except Exception as e:
        logging.error(f"Error fetching email {num} from {folder}: {e}")
        return None


def fetch_emails(start_date, end_date, access_token, max_emails=100):
    """
    Enhanced email fetching with robust IMAP connection handling and last fetch tracking.
    Only fetches emails newer than the last fetch time for each folder, for fast dashboard updates.
    """
    all_email_data = []
    messages = []

    for folder in EMAIL_FOLDERS:
        mail = None
        try:
            # Get fresh IMAP connection for each folder
            mail, conn_error = get_imap_connection(access_token, folder)
            if not mail:
                messages.append(
                    f"Connection failed for folder {folder}: {conn_error}")
                logging.warning(
                    f"Skipping folder {folder} due to connection failure: {conn_error}")
                continue

            # Use last fetch time if available
            last_fetch_time = get_last_fetch_time(folder)
            if last_fetch_time:
                fetch_start_date = datetime.strptime(
                    last_fetch_time, "%Y-%m-%d %H:%M:%S")
            else:
                fetch_start_date = start_date
            start_date_str = fetch_start_date.strftime("%d-%b-%Y")
            email_data = []

            # Process each sender sequentially to avoid OAuth2 conflicts
            for sender in SENDERS:
                try:
                    result, data = mail.search(
                        None, f'SINCE {start_date_str} FROM "{sender}"')
                    if result != "OK":
                        messages.append(
                            f"Error searching emails from {sender} in folder {folder}: {result}")
                        continue

                    mail_ids = data[0].split()
                    if not mail_ids:
                        continue

                    # Limit emails per sender to avoid overwhelming the system
                    mail_ids = mail_ids[-max_emails:] if len(
                        mail_ids) > max_emails else mail_ids

                    # Process emails sequentially to avoid connection issues
                    for num in reversed(mail_ids):
                        try:
                            email_result = fetch_email_direct(
                                mail, num, folder)
                            if email_result:
                                # Only add emails newer than last_fetch_time (if available)
                                email_date = pd.to_datetime(
                                    email_result[0], errors='coerce')
                                if last_fetch_time:
                                    if email_date > fetch_start_date:
                                        email_data.append(email_result)
                                else:
                                    email_data.append(email_result)
                        except Exception as email_error:
                            logging.warning(
                                f"Failed to fetch email {num} from {sender}: {email_error}")
                            continue

                    logging.info(
                        f"Processed {len(mail_ids)} emails from {sender} in {folder}")

                except Exception as sender_error:
                    logging.error(
                        f"Error processing sender {sender} in folder {folder}: {sender_error}")
                    continue

            all_email_data.extend(email_data)
            messages.append(
                f"Fetched {len(email_data)} emails from folder {folder}")

            # Update last fetch time for this folder
            update_last_fetch_time(
                folder, end_date.strftime("%Y-%m-%d %H:%M:%S"))

        except Exception as e:
            error_msg = f"Error fetching emails from folder {folder}: {e}"
            messages.append(error_msg)
            logging.error(error_msg)

        finally:
            # Always clean up the connection
            if mail:
                try:
                    mail.close()
                    mail.logout()
                    logging.info(
                        f"Successfully closed connection to folder {folder}")
                except Exception as cleanup_error:
                    logging.warning(
                        f"Error during connection cleanup for {folder}: {cleanup_error}")

    # Process results
    if not all_email_data:
        messages.append("No emails found matching the criteria")
        return pd.DataFrame(columns=["Date", "Sender", "Subject", "Mail Body", "Folder"]), messages

    df = pd.DataFrame(all_email_data, columns=[
                      "Date", "Sender", "Subject", "Mail Body", "Folder"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna(df["Date"])

    messages.append(f"Total emails fetched across all folders: {len(df)}")
    logging.info(
        f"Successfully fetched {len(df)} emails across {len(EMAIL_FOLDERS)} folders")

    return df, messages


def process_and_store_emails(df):
    """Process and store emails in the enhanced database"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    email_records = []
    for _, row in df.iterrows():
        subject = row["Subject"]
        sender = row["Sender"]
        date = row["Date"]
        mail_body = row["Mail Body"]
        folder = row["Folder"]

        # Detect critical keywords and severity
        critical_keywords, severity = detect_critical_keywords(
            subject + " " + mail_body)

        # Extract application name
        app_name = extract_application_name(subject, sender, mail_body)

        # Generate unique ID and hash
        email_id = str(uuid.uuid5(uuid.NAMESPACE_DNS,
                       f"{date}_{sender}_{subject[:50]}"))
        email_hash = hashlib.md5(
            f"{date}_{sender}_{subject}_{mail_body[:100]}".encode()).hexdigest()

        email_records.append((
            email_id, date, sender, subject, mail_body, folder,
            severity, ",".join(critical_keywords) if critical_keywords else "",
            app_name, email_hash
        ))

    try:
        cursor.executemany('''INSERT OR REPLACE INTO emails
                             (id, date, sender, subject, body, folder,
                              severity, keywords, application, hash)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', email_records)
        conn.commit()
        logging.info(
            f"Successfully stored {len(email_records)} emails in database")
    
    except Exception as e:
        logging.error(f"Failed to insert emails into database: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

# Helper functions for email processing


def detect_critical_keywords(text):
    """Detect critical keywords in text and return keywords with severity"""
    if not text:
        return [], "LOW"

    text_lower = text.lower()
    found_keywords = []
    highest_severity = "LOW"

    # Check keywords in order of severity (CRITICAL first, then HIGH, etc.)
    severity_order = ["HIGH", "MEDIUM", "LOW"]

    for severity in severity_order:
        keywords = CRITICAL_KEYWORDS.get(severity, [])
        for keyword in keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
                # Update highest severity if we found a more critical keyword
                if severity_order.index(severity) < severity_order.index(highest_severity):
                    highest_severity = severity

    return list(set(found_keywords)), highest_severity


def extract_application_name(subject, sender=None, mail_body=None):
    """Extract application name from email data"""
    # Enhanced patterns for application names
    patterns = [
        # Critical:/HYD_SITE_DOMAIN -> HYD_SITE_DOMAIN
        r'Critical:/?([A-Z_]+)',
        r'Alert:/?([A-Z_]+)',     # Alert:/OCI_DOMAIN -> OCI_DOMAIN
        r'Error:/?([A-Z_]+)',     # Error:/WEBLOGIC_DOMAIN -> WEBLOGIC_DOMAIN
        r'([A-Z_]+)_DOMAIN',      # Any _DOMAIN suffix
        r'([A-Z_]+)_SITE',        # Any _SITE suffix
        r'([A-Z_]+)_PROD',        # Any _PROD suffix
        r'([A-Z_]+)_DEV',         # Any _DEV suffix
        r'([A-Z_]+)_ALERT',       # Any _ALERT suffix
        r'([A-Z_]+)_ERROR',       # Any _ERROR suffix
        r'([A-Z_]+)_FAILURE',     # Any _FAILURE suffix
        r'([A-Z_]+)_DOWN',        # Any _DOWN suffix
        r'([A-Z_]+)_TIMEOUT',     # Any _TIMEOUT suffix
    ]

    for pattern in patterns:
        match = re.search(pattern, subject, re.IGNORECASE)
        if match:
            app_name = match.group(1)
            # Clean up the application name
            app_name = app_name.replace('_', ' ').title()
            return app_name

    # Enhanced keyword detection for specific applications
    subject_upper = subject.upper()
    subject_lower = subject.lower()

    # Banking and Financial Applications
    for keyword in ['AXIOM', 'AXIOM_', 'AXIOM-']:
        if keyword in subject_upper:
            return 'AXIOM'
    for keyword in ['Anamoly', 'ANAMOLY', 'Anamoly-']:
        if keyword in subject_upper:
            return 'UPI'
    for keyword in ['Falcon', 'FALCON', 'DDA']:
        if keyword in subject_upper:
            return 'Falcon'
    for keyword in ['Control M', 'Control_', 'Control-']:
        if keyword in subject_upper:
            return 'Control M'
    for keyword in ['ALM', 'ALM_', 'ALM-']:
        if keyword in subject_upper:
            return 'ALM'
    for keyword in ['DLMS', 'DLMS_', 'DLMS-']:
        if keyword in subject_upper:
            return 'DLMS'
    for keyword in ['AMLCK', 'AMLCK_', 'AMLCK&lt']:
        if keyword in subject_upper:
            return 'AMLCK'
    for keyword in ['KMT', 'KMT_', 'KMT-']:
        if keyword in subject_upper:
            return 'KMT'
    for keyword in ['OBIEE', 'OBIEE_', 'OBIEE-']:
        if keyword in subject_upper:
            return 'OBIEE'
    for keyword in ['KARIX', 'KARIX_', 'KARIX-']:
        if keyword in subject_upper:
            return 'Karix'
    for keyword in ['NPA', 'Enterprise NPA System', 'NPA-']:
        if keyword in subject_upper:
            return 'NPA System'
    for keyword in ['CredPro', 'CREDPRO', 'CredPro&lt']:
        if keyword in subject_upper:
            return 'CredPro'
    for keyword in ['FIG', 'FIG-', '-FIG']:
        if keyword in subject_upper:
            return 'FIG'
    for keyword in ['CTS', 'CTS-', '-CTS']:
        if keyword in subject_upper:
            return 'CTS'
    for keyword in ['CBS', 'CBS_', 'CBS-', 'CORE_BANKING']:
        if keyword in subject_upper:
            return 'CBS (Core Banking)'
    for keyword in ['ODI', 'ODI_', 'ODI-', 'ORACLE_DATA_INTEGRATOR']:
        if keyword in subject_upper:
            return 'ODI (Oracle Data Integrator)'
    for keyword in ['IVR', 'IVR_', 'IVR-', 'INTERACTIVE_VOICE']:
        if keyword in subject_upper:
            return 'IVR (Interactive Voice Response)'
    for keyword in ['CIBIL', 'CIBIL_', 'CIBIL-', 'CREDIT_BUREAU']:
        if keyword in subject_upper:
            return 'CIBIL (Credit Bureau)'
    for keyword in ['JANSURAKSHA', 'JAN_SURAKSHA', 'JAN-SURAKSHA']:
        if keyword in subject_upper:
            return 'Jansuraksha'
    for keyword in ['VKYC', 'V_KYC', 'V-KYC', 'VIDEO_KYC']:
        if keyword in subject_upper:
            return 'VKYC (Video KYC)'
    for keyword in ['CC_ONBOARDING', 'CC-ONBOARDING', 'CREDIT_CARD_ONBOARDING']:
        if keyword in subject_upper:
            return 'CC Onboarding'
    for keyword in ['EXPERIAN_BUREAU_SELF', 'EXPERIAN_BUREAU', 'EXPERIAN']:
        if keyword in subject_upper:
            return 'Experian Bureau Self'

    # Additional Banking and Financial Applications
    for keyword in ['IDAM', 'IDAM_', 'IDAM-']:
        if keyword in subject_upper:
            return 'IDAM'
    for keyword in ['EWS', 'EWS_', 'EWS-']:
        if keyword in subject_upper:
            return 'EWS'
    for keyword in ['FICO', 'Fico', 'FICO_']:
        if keyword in subject_upper:
            return 'FICO'
    for keyword in ['SAPPHIREIMS', 'SAPPHIRE_IMS', 'SAPPHIRE-IMS']:
        if keyword in subject_upper:
            return 'SapphireIMS'
    for keyword in ['SEWA', 'SEWA_', 'SEWA-']:
        if keyword in subject_upper:
            return 'SEWA'
    for keyword in ['CIB', 'CIB_', 'CIB-']:
        if keyword in subject_upper:
            return 'CIB'
    for keyword in ['KARZA', 'KARZA_', 'KARZA-']:
        if keyword in subject_upper:
            return 'Karza'
    for keyword in ['CIMS', 'CIMS_ADF', 'CIMS-ADF', 'CIMS-ADF']:
        if keyword in subject_upper:
            return 'CIMS-ADF'
    for keyword in ['POSIDEX', 'POSIDEX_', 'POSIDEX-']:
        if keyword in subject_upper:
            return 'Posidex'
    for keyword in ['VIDEO_BANKING', 'VIDEOBANKING', 'VIDEO-BANKING']:
        if keyword in subject_upper:
            return 'Video Banking'
    for keyword in ['CKYC', 'CKYC_', 'CKYC-']:
        if keyword in subject_upper:
            return 'CKYC'
    for keyword in ['BIMS', 'BIMS_', 'BIMS-']:
        if keyword in subject_upper:
            return 'BIMS'
    for keyword in ['NACH', 'NACH_', 'NACH-']:
        if keyword in subject_upper:
            return 'NACH'
    for keyword in ['PFMS', 'PFMS_', 'PFMS-']:
        if keyword in subject_upper:
            return 'PFMS'
    for keyword in ['EGOVPAY', 'EGOV_PAY', 'EGOV-PAY']:
        if keyword in subject_upper:
            return 'EGovPay'
    for keyword in ['RAPID', 'RAPID_', 'HOTFOOT', '-HOTFOOT']:
        if keyword in subject_upper:
            return 'RAPID-HOTFOOT'
    for keyword in ['CRM', 'CRM_', 'CRM&lt', ';CRM&lt;']:
        if keyword in subject_upper:
            return 'CRM'

    # Infrastructure and Monitoring Applications
    for keyword in ['OCI', 'ORACLE_CLOUD']:
        if keyword in subject_upper:
            return 'OCI (Oracle Cloud)'
    for keyword in ['ESB', 'ESB_']:
        if keyword in subject_upper:
            return 'ESB'
    for keyword in ['ORACLE', 'ORACLE_DB']:
        if keyword in subject_upper:
            return 'Oracle Database'
    for keyword in ['FRESHPING', 'FRESH_PING']:
        if keyword in subject_upper:
            return 'FreshPing'
    for keyword in ['EURONET', 'EURO_NET']:
        if keyword in subject_upper:
            return 'Euronet'

    # Additional Banking Systems
    for keyword in ['ATM', 'ATM_', 'ATM-']:
        if keyword in subject_upper:
            return 'ATM System'
    for keyword in ['SWITCH', 'SWITCH_', 'SWITCH-']:
        if keyword in subject_upper:
            return 'Switch System'
    for keyword in ['CARD', 'CARD_', 'CARD-']:
        if keyword in subject_upper:
            return 'Credit Card'
    for keyword in ['PAYMENT', 'PAYMENT_', 'PAYMENT-']:
        if keyword in subject_upper:
            return 'Payment Gateway'
    for keyword in ['MOBILE', 'MOBILE_', 'MOBILE-']:
        if keyword in subject_upper:
            return 'Mobile Banking'
    for keyword in ['INTERNET', 'INTERNET_', 'INTERNET-']:
        if keyword in subject_upper:
            return 'Internet Banking'

    # Network and Infrastructure
    for keyword in ['NETWORK', 'NETWORK_', 'NETWORK-']:
        if keyword in subject_upper:
            return 'Network Infrastructure'
    for keyword in ['DATABASE', 'DB_', 'DB-']:
        if keyword in subject_upper:
            return 'Database System'

    for keyword in ['MIDDLEWARE', 'MIDDLE_WARE']:
        if keyword in subject_upper:
            return 'Middleware'

    # Try to extract from domain patterns
    domain_patterns = [
        r'([A-Z]+)_SITE',
        r'([A-Z]+)_DOMAIN',
        r'([A-Z]+)_ENV',
        r'([A-Z]+)_INSTANCE',
        r'([A-Z]+)_SERVICE',
    ]

    for pattern in domain_patterns:
        match = re.search(pattern, subject, re.IGNORECASE)
        if match:
            app_name = match.group(1)
            app_name = app_name.replace('_', ' ').title()
            return app_name

    # If no application detected, use sender email domain as fallback
    if sender:
        # First check for specific sender email patterns
        sender_lower = sender.lower()
        if 'nttin.hnoc2.support@global.ntt' in sender_lower or 'nttin.alerts@global.ntt' in sender_lower:
            return 'Netmagic'
        elif 'emr_support@euronetworldwide.com' in sender_lower:
            return 'EMR'
        elif 'eurodesk@euronetworldwide.com' in sender_lower or 'enupisupport@euronetworldwide.com' in sender_lower:
            return 'EURONET'

        elif 'noreply@alerts.elastic.co' in sender_lower or 'noreply@alerts.elastic.co' in sender_lower:
            return 'FINCARE'

        # Then check domain patterns
        domain_match = re.search(r'@([^.]+)', sender)
        if domain_match:
            domain = domain_match.group(1).upper()
            # Map common domains to readable names
            domain_mapping = {
                'EURONETWORLDWIDE': 'Euronet System',
                'GLOBAL': 'Global System',
                'ACLMOBILEALERT': 'ACL Mobile System',
                'QUALITYKIOSK': 'QualityKiosk System',
                'KARIX': 'Karix System',
                'NEWRELIC': 'New Relic System',
                'AMAZONAWS': 'AWS System',
                'FRESHPING': 'FreshPing System',
                'NTTSERVICES': 'NTT System'
            }
            return domain_mapping.get(domain, f"{domain.title()} System")

    # If all else fails, fall back to the original method
    text_to_search = f"{subject} {sender or ''} {mail_body or ''}"
    app_patterns = [
        r'oracle|weblogic|websphere|tomcat|apache|nginx',
        r'mysql|postgresql|sqlserver|mongodb|redis',
        r'aws|azure|gcp|cloud',
        r'vmware|hyperv|kvm',
        r'windows|linux|unix',
        r'network|firewall|router|switch',
        r'email|smtp|imap|pop3',
        r'database|db|sql',
        r'web|http|https|api',
        r'backup|storage|disk|file'
    ]

    for pattern in app_patterns:
        match = re.search(pattern, text_to_search, re.IGNORECASE)
        if match:
            return match.group(0).title()

    return "Unknown"


def ensure_qdrant_client_open():
    """Ensure the Qdrant client is open and working with better reconnection logic"""
    if 'vector_db_manager' not in st.session_state:
        return
        
    if not hasattr(st.session_state.vector_db_manager, 'client'):
        return
        
    try:
        # Test the client with a simple operation
        st.session_state.vector_db_manager.client.get_collections()
    except Exception as e:
        logging.warning(f"Qdrant client reconnection needed: {e}")
        try:
            # Create new client and reconnect
            st.session_state.vector_db_manager.client = QdrantClient(
                host=QDRANT_HOST, port=QDRANT_PORT, timeout=10.0)  # Add timeout parameter
            
            # Skip if embeddings not initialized
            if not hasattr(st.session_state.vector_db_manager, 'embeddings') or \
               st.session_state.vector_db_manager.embeddings is None:
                return
                
            # Reconnect vector_db
            st.session_state.vector_db_manager.vector_db = Qdrant(
                client=st.session_state.vector_db_manager.client,
                collection_name=QDRANT_COLLECTION_NAME,
                embeddings=st.session_state.vector_db_manager.embeddings
            )
            logging.info("Qdrant client reconnected successfully")
            
            # Verify connection was successful with a test query
            test_result = st.session_state.vector_db_manager.client.get_collections()
            if test_result:
                logging.info(f"Qdrant reconnection verified with {len(test_result.collections)} collections")
        except Exception as reconnect_error:
            logging.error(f"Failed to reconnect Qdrant: {reconnect_error}")
            st.session_state.vector_db_manager.vector_db = None  # Reset to avoid using broken connection
            
# UI Rendering Functions
def render_email_list():
    """Render the recent emails list with improved time display"""
    st.markdown("<h2 class='section-header'>📧 Recent Emails</h2>",
                unsafe_allow_html=True)
    try:
        conn = sqlite3.connect(DB_FILE)

        # Show current time for reference
        now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
        st.info(f"📅 Current Time (IST): {now_ist}")

        # Get the most recent emails with better date formatting
        recent_emails = pd.read_sql_query('''
            SELECT
                date,
                datetime(date) as formatted_date,
                sender,
                subject,
                severity,
                application,
                keywords
            FROM emails
            ORDER BY date DESC
            LIMIT 50
        ''', conn)

        if not recent_emails.empty:
            # Calculate time ago for better context
            recent_emails['date'] = pd.to_datetime(recent_emails['date'])
            # Remove IST timezone to avoid tz-aware vs tz-naive comparison
            now = pd.Timestamp(datetime.now())

            # Create a human-readable "time ago" column
            def time_ago(timestamp):
                # Ensure both timestamps are tz-naive for comparison
                if timestamp.tzinfo is not None:
                    timestamp = timestamp.tz_localize(None)

                diff = now - timestamp
                seconds = diff.total_seconds()

                if seconds < 60:
                    return f"{int(seconds)} seconds ago"
                if seconds < 3600:
                    return f"{int(seconds/60)} minutes ago"
                if seconds < 86400:
                    return f"{int(seconds/3600)} hours ago"
                if seconds < 604800:
                    return f"{int(seconds/86400)} days ago"
                return f"{int(seconds/604800)} weeks ago"

            recent_emails['time_ago'] = recent_emails['date'].apply(time_ago)

            # Display dataframe with time ago information
            # Use a unique key to prevent duplicate view errors
            st.dataframe(
                recent_emails[['formatted_date', 'time_ago', 'sender', 'subject', 'severity', 'application', 'keywords']],
                use_container_width=True,
                column_config={
                    "formatted_date": st.column_config.DatetimeColumn("Date"),
                    "time_ago": st.column_config.TextColumn("Time Ago"),
                    "sender": st.column_config.TextColumn("Sender"),
                    "subject": st.column_config.TextColumn("Subject"),
                    "severity": st.column_config.SelectboxColumn("Severity", options=["CRITICAL", "HIGH", "MEDIUM", "LOW"]),
                    "application": st.column_config.TextColumn("Application"),
                    "keywords": st.column_config.TextColumn("Keywords")
                },
                key="recent_emails_table"
            )
        else:
            st.info(
                "No emails found. Click 'Fetch New Emails Now' in the sidebar to populate the database.")

        conn.close()
    except Exception as e:
        st.error(f"Failed to load recent emails: {e}")
        logging.error(f"Email list render error: {str(e)}")


def render_settings():
    """Render the settings interface"""
    st.markdown("<h2 class='section-header'>⚙️ System Settings</h2>",
                unsafe_allow_html=True)
    try:
        # Database information
        st.markdown("### Database Information")
        conn = sqlite3.connect(DB_FILE)

        # Table sizes
        tables = ['emails', 'alerts', 'llm_interactions', 'incident_reports']
        for table in tables:
            count = pd.read_sql_query(
                f'SELECT COUNT(*) as count FROM {table}', conn).iloc[0]['count']
            st.metric(f"{table.title()} Count", count)

        # Database size
        db_size = os.path.getsize(DB_FILE) / (1024 * 1024)  # MB
        st.metric("Database Size", f"{db_size:.2f} MB")

        conn.close()

        # LLM Configuration
        st.markdown("### LLM Configuration")
        if TINYLLAMA_AVAILABLE:
            st.success("✅ TinyLlama is available!")
            st.info("Model: tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
        else:
            st.warning("⚠️ TinyLlama not available")
            if st.button("📥 Setup TinyLlama", key="btn_setup_tinyllama"):
                st.info("1. Run: python3 setup_tinyllama.py")
                st.info(
                    "2. Download from: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
                st.info("3. Place in 'models' directory")


        # Vector Database Settings
        st.markdown("### Vector Database Settings")
        if st.button("🔄 Rebuild Vector Database", key="btn_rebuild_vector_db"):  # Added key
            with st.spinner("Rebuilding vector database..."):
                if 'vector_db_manager' in st.session_state:
                    st.session_state.vector_db_manager = VectorDBManager()
                    st.success("Vector database rebuilt successfully!")

        # Data Retention
        st.markdown("### Data Retention")
        if st.button("🧹 Clean Old Data", key="btn_clean_old_data"):  # Added key
            cleanup_old_data()
            st.success("Old data cleaned successfully!")

    except Exception as e:
        st.error(f"Failed to render settings: {e}")
        logging.error(f"Settings render error: {traceback.format_exc()}")

# LLM Query Processing


class LLMQueryProcessor:

    def __init__(self, llm_config: LLMConfig, vector_db_manager: VectorDBManager):
        self.llm_config = llm_config
        self.vector_db_manager = vector_db_manager
        # Updated memory initialization
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        # Instead of accessing chat_memory.messages directly
        if hasattr(self.memory, 'chat_memory') and hasattr(self.memory.chat_memory, 'messages'):
            pass  # The memory is properly initialized
        self.response_cache = {}  # Simple cache for responses
        self.current_model_type = "fast"  # Default to fast model
        self.llm = None  # Will be initialized when needed

    def get_llm(self, model_type="fast"):
        """Get LLM instance with specified model type"""
        if self.llm is None or self.current_model_type != model_type:
            self.current_model_type = model_type
            if TINYLLAMA_AVAILABLE:
                self.llm = self.llm_config.get_llm(model_type)
            else:
                self.llm = self.llm_config.get_llm()
        return self.llm

    def create_qa_chain(self, model_type="fast"):
        """Create a question-answering chain"""
        llm = self.get_llm(model_type)
        if not llm:
            return None

        if not self.vector_db_manager.vector_db:
            st.warning("Vector database not available - using direct LLM query")
            return self.create_direct_qa_chain(llm)

        try:
            # Ultra-fast prompt template for better performance
            template = """IT Alert Analysis - Quick Response:

            Context: {context}
            Question: {question}

            Quick Answer:"""

            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )

            # Create retrieval chain with optimized settings
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vector_db_manager.vector_db.as_retriever(
                    search_kwargs={"k": 2}),  # Reduced from 3 to 2
                chain_type_kwargs={"prompt": prompt}
            )

            return qa_chain

        except Exception as e:
            st.error(f"Failed to create QA chain: {e}")
            st.info("Falling back to direct LLM query...")
            return self.create_direct_qa_chain(llm)

    def create_direct_qa_chain(self, llm):
        """Create a direct QA chain without vector database"""
        try:
            template = """You are an IT operations expert analyzing system alerts and emails.

Question: {question}

Based on your knowledge of IT systems, provide a helpful analysis and recommendations.
Focus on:
1. Identifying the type of issue
2. Suggesting potential causes
3. Recommending immediate actions
4. Providing preventive measures

Answer:"""

            prompt = PromptTemplate(
                template=template,
                input_variables=["question"]
            )

            from langchain.chains import LLMChain
            return LLMChain(llm=llm, prompt=prompt)

        except Exception as e:
            st.error(f"Failed to create direct QA chain: {e}")
            return None

    def process_query(self, query: str, model_type="fast") -> Dict[str, Any]:
        """Process a user query using LLM"""
        try:
            # Check cache first for faster responses
            cache_key = f"{query}_{model_type}"
            if CACHE_LLM_RESPONSES and cache_key in self.response_cache:
                st.info("📋 Using cached response for faster results")
                return self.response_cache[cache_key]

            llm = self.get_llm(model_type)
            if not llm:
                # Fallback: Provide similar emails without LLM analysis
                similar_emails = self.vector_db_manager.search_similar_emails(
                    query, k=3)

                if similar_emails:
                    context = "Based on similar emails found in the system:\n\n"
                    for doc, score in similar_emails:
                        context += f"• Email (similarity: {score:.3f}): {doc.page_content}\n\n"

                    fallback_response = f"""
                    **LLM Analysis Unavailable**

                    I found {len(similar_emails)} similar emails related to your query.
                    Here's what I found:

                    {context}

                    **To enable AI analysis:**
                    1. Download TinyLlama: `python3 setup_tinyllama.py`
                    2. Or manually download from: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
                    3. Place the model files in the 'models' directory
                    4. Restart the application
                    """

                    return {
                        "response": fallback_response,
                        "context_emails": similar_emails,
                        "similarity_scores": [score for _, score in similar_emails],
                        "mode": "fallback"
                    }
                else:
                    return {
                        "error": "LLM not available and no similar emails found. Please download TinyLlama to enable AI analysis."
                    }

            # Search for similar emails (ultra-reduced for faster processing)
            similar_emails = self.vector_db_manager.search_similar_emails(
                query, k=2)  # Reduced to 2 for ultra-fast mode

            # Create minimal context for faster processing
            context = ""
            for doc, score in similar_emails:
                content = doc.page_content[:200] if FAST_MODE else doc.page_content[:500]
                context += f"Email ({score:.2f}): {content}\n"

            # Create QA chain
            qa_chain = self.create_qa_chain(model_type)
            if not qa_chain:
                return {"error": "Failed to create QA chain"}

            # Get response
            if hasattr(qa_chain, 'invoke'):
                # Use the newer invoke method
                response = qa_chain.invoke({"query": query})
                if isinstance(response, dict):
                    response = response.get('result', response.get('text', str(response)))
            elif hasattr(qa_chain, 'run'):
                # Fallback to run for compatibility
                response = qa_chain.run(query)
            else:
                # Direct LLM chain
                response = qa_chain({"question": query})
                if isinstance(response, dict):
                    response = response.get('text', str(response))

            # Store interaction
            store_llm_interaction(query, response, context)

            result = {
                "response": response,
                "context_emails": similar_emails,
                "similarity_scores": [score for _, score in similar_emails],
                "mode": "llm",
                "model_type": model_type
            }

            # Cache the result for faster future queries
            if CACHE_LLM_RESPONSES:
                self.response_cache[cache_key] = result

            return result

        except Exception as e:
            st.error(f"Failed to process query: {e}")
            return {"error": str(e)}


def store_llm_interaction(query: str, response: str, context: str):
    """Store LLM interaction in database"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Convert context to string if it's not already a string
        if not isinstance(context, str):
            context = str(context)

        interaction_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO llm_interactions (id, query, response, context_emails)
            VALUES (?, ?, ?, ?)
        ''', (interaction_id, query, response, context))

        conn.commit()
        conn.close()

    except Exception as e:
        st.error(f"Failed to store LLM interaction: {e}")


def fast_keyword_analysis(query: str) -> Dict[str, Any]:
    """Enhanced fast keyword-based analysis without LLM"""
    try:
        # Extract keywords from query
        query_lower = query.lower()
        found_keywords = []

        # Check for critical keywords in query
        for severity, keywords in CRITICAL_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    found_keywords.append((keyword, severity))

        # Also extract application names from query for better relevance
        potential_app_name = extract_application_name(query)
        
        # Search database for emails with similar keywords or applications
        conn = sqlite3.connect(DB_FILE)

        # Build SQL query based on found keywords and applications
        conditions = []
        params = []
        
        # Add keyword conditions
        for keyword, _ in found_keywords:
            conditions.append("(keywords LIKE ? OR subject LIKE ? OR body LIKE ?)")
            keyword_param = f"%{keyword}%"
            params.extend([keyword_param, keyword_param, keyword_param])
            
        # Add application condition if found
        if potential_app_name and potential_app_name != "Unknown":
            conditions.append("(application LIKE ?)")
            params.append(f"%{potential_app_name}%")
            
        # Add general query terms for broader matching
        query_terms = [term for term in query_lower.split() if len(term) > 2]
        for term in query_terms:
            conditions.append("(subject LIKE ? OR body LIKE ?)")
            term_param = f"%{term}%"
            params.extend([term_param, term_param])
            
        # Combine all conditions with OR
        if conditions:
            where_clause = " OR ".join(conditions)
            sql_query = f'''
                SELECT date, sender, subject, severity, keywords, application, body
                FROM emails
                WHERE {where_clause}
                ORDER BY date DESC
                LIMIT 10
            '''
        else:
            # Fallback if no conditions were created
            sql_query = '''
                SELECT date, sender, subject, severity, keywords, application, body
                FROM emails
                ORDER BY date DESC
                LIMIT 5
            '''
            params = []

        similar_emails = pd.read_sql_query(sql_query, conn, params=params)
        conn.close()

        # Generate enhanced response based on found emails
        if not similar_emails.empty:
            # Count by severity
            severity_counts = similar_emails['severity'].value_counts()
            
            # Extract snippets from body for context
            similar_emails['snippet'] = similar_emails['body'].apply(
                lambda x: x[:150] + "..." if isinstance(x, str) and len(x) > 150 else x
            )

            response = f"""
**Fast Analysis Results for: "{query}"**

**Found Keywords:** {', '.join([kw for kw, _ in found_keywords]) if found_keywords else 'None detected'}
**Detected Application:** {potential_app_name if potential_app_name != "Unknown" else "None detected"}

**Alert Summary:**
- Total related emails found: {len(similar_emails)}
- High priority alerts: {severity_counts.get('HIGH', 0)}
- Medium priority alerts: {severity_counts.get('MEDIUM', 0)}
- Low priority alerts: {severity_counts.get('LOW', 0)}

**Top Applications Affected:**
{similar_emails['application'].value_counts().head(3).to_string() if not similar_emails['application'].empty else 'None identified'}

**Recent Related Alerts:**
"""
            # Add top 3 emails with snippets
            for i, row in similar_emails.head(3).iterrows():
                response += f"\n{row['date']} | {row['subject']} | {row['severity']}\nSnippet: {row['snippet']}\n"
                
        else:
            response = f"""
**Fast Analysis Results for: "{query}"**

No related emails found in the database. This could mean:
- The issue is new and hasn't been reported yet
- The search terms don't match existing alerts
- Try using different keywords or check the "Ultra Fast" mode for similar emails
            """

        return {
            "response": response,
            "similar_emails": similar_emails.drop(columns=['body']).to_dict('records') if not similar_emails.empty else [],
            "found_keywords": found_keywords,
            "mode": "fast_keyword"
        }

    except Exception as e:
        return {
            "response": f"Fast analysis failed: {str(e)}",
            "similar_emails": [],
            "found_keywords": [],
            "mode": "error"
        }
    

def enhanced_ultra_fast_analysis(query: str, vector_db_manager: VectorDBManager) -> Dict[str, Any]:
    """Enhanced ultra-fast analysis with structured output"""
    try:
        # Get similar emails
        similar_emails = vector_db_manager.search_similar_emails(query, k=5)
        
        if not similar_emails:
            # Try keyword fallback
            similar_emails = vector_db_manager.keyword_search_fallback(query, k=5)
        
        if similar_emails:
            # Analyze patterns in similar emails
            subjects = []
            severities = []
            applications = []
            keywords_found = []
            
            for doc, score in similar_emails:
                if hasattr(doc, 'metadata'):
                    subjects.append(doc.metadata.get('subject', ''))
                    severities.append(doc.metadata.get('severity', 'UNKNOWN'))
                    applications.append(doc.metadata.get('application', 'Unknown'))
                
                # Extract keywords from content
                content = doc.page_content.lower()
                for severity, keywords in CRITICAL_KEYWORDS.items():
                    for keyword in keywords:
                        if keyword.lower() in content:
                            keywords_found.append((keyword, severity))
            
            # Generate structured analysis
            severity_counts = pd.Series(severities).value_counts()
            app_counts = pd.Series([app for app in applications if app != 'Unknown']).value_counts()
            
            response = f"""
## 🚀 Ultra-Fast Analysis Results

**Query:** "{query}"

### 📊 Quick Summary
- **Similar incidents found:** {len(similar_emails)}
- **Most common severity:** {severity_counts.index[0] if not severity_counts.empty else 'Unknown'}
- **Top affected application:** {app_counts.index[0] if not app_counts.empty else 'Unknown'}

### 🔥 Critical Patterns Detected
"""
            
            # Add top keywords found
            if keywords_found:
                keyword_severity = {}
                for kw, sev in keywords_found:
                    if sev not in keyword_severity:
                        keyword_severity[sev] = []
                    keyword_severity[sev].append(kw)
                
                for severity in ['HIGH', 'MEDIUM', 'LOW']:
                    if severity in keyword_severity:
                        response += f"- **{severity}:** {', '.join(set(keyword_severity[severity]))}\n"
            
            response += f"""
### 🎯 Immediate Actions Recommended
"""
            
            # Generate recommendations based on patterns
            if 'HIGH' in severity_counts and severity_counts['HIGH'] > 0:
                response += "- 🚨 **URGENT**: High-severity patterns detected - immediate attention required\n"
            
            if app_counts.empty == False:
                top_app = app_counts.index[0]
                response += f"- 🔍 **Focus on {top_app}**: Most affected application in similar incidents\n"
            
            response += f"- 📞 **Check recent alerts**: Review last {len(similar_emails)} similar incidents for patterns\n"
            response += "- 📋 **Monitor**: Set up monitoring for detected keywords\n"
            
            return {
                "response": response,
                "similar_emails": similar_emails,
                "severity_analysis": severity_counts.to_dict(),
                "application_analysis": app_counts.to_dict(),
                "keywords_found": keywords_found,
                "mode": "enhanced_ultra_fast"
            }
        else:
            return {
                "response": """
## 🚀 Ultra-Fast Analysis Results

**No similar incidents found** for your query. This could indicate:

- ✅ **New Issue**: This might be a new type of problem
- 🔍 **Different Keywords**: Try rephrasing your query
- 📊 **Check Recent Data**: Ensure recent emails have been fetched

### 🎯 Recommended Actions
- Use different search terms
- Check if the issue is currently happening
- Review recent email fetches in Settings tab
""",
                "similar_emails": [],
                "mode": "enhanced_ultra_fast"
            }
    except Exception as e:
        return {
            "response": f"Analysis failed: {str(e)}",
            "mode": "error"
        }
    

def enhanced_fast_keyword_analysis(query: str) -> Dict[str, Any]:
    """Enhanced fast keyword analysis with actionable insights"""
    try:
        query_lower = query.lower()
        found_keywords = []
        
        # Enhanced keyword detection
        for severity, keywords in CRITICAL_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    found_keywords.append((keyword, severity))
        
        # Extract application from query
        potential_app = extract_application_name(query)
        
        # Database analysis with better SQL
        conn = sqlite3.connect(DB_FILE)
        
        # Build comprehensive query
        conditions = []
        params = []
        
        # Add keyword conditions
        for keyword, severity in found_keywords:
            conditions.append("(keywords LIKE ? OR subject LIKE ? OR body LIKE ?)")
            keyword_param = f"%{keyword}%"
            params.extend([keyword_param, keyword_param, keyword_param])
        
        # Add application condition
        if potential_app and potential_app != "Unknown":
            conditions.append("application LIKE ?")
            params.append(f"%{potential_app}%")
        
        # Add general query terms
        query_terms = [term for term in query_lower.split() if len(term) > 2]
        for term in query_terms:
            conditions.append("(subject LIKE ? OR body LIKE ?)")
            term_param = f"%{term}%"
            params.extend([term_param, term_param])
        
        if conditions:
            where_clause = " OR ".join(conditions)
            sql_query = f'''
                SELECT date, sender, subject, severity, keywords, application, body,
                       CASE 
                           WHEN severity = 'HIGH' THEN 3
                           WHEN severity = 'MEDIUM' THEN 2
                           WHEN severity = 'LOW' THEN 1
                           ELSE 0
                       END as severity_score
                FROM emails
                WHERE {where_clause} AND date >= datetime('now', '-7 days')
                ORDER BY severity_score DESC, date DESC
                LIMIT 20
            '''
            
            related_emails = pd.read_sql_query(sql_query, conn, params=params)
        else:
            related_emails = pd.DataFrame()
        
        conn.close()
        
        if not related_emails.empty:
            # Advanced analysis
            severity_dist = related_emails['severity'].value_counts()
            severity_breakdown = {k: int(v) for k, v in severity_dist.items()}
            app_dist = related_emails['application'].value_counts()
            recent_trend = related_emails.head(10)
            
            # Time-based analysis
            related_emails['date'] = pd.to_datetime(related_emails['date'], errors='coerce')
            now_dt = pd.to_datetime(datetime.now())
            last_24h = related_emails[related_emails['date'] > (now_dt - timedelta(days=1))]
            
            response = f"""
### 📊 Historical Pattern Analysis
- **Total related incidents:** {len(related_emails)}
- **Last 24 hours:** {len(last_24h)} incidents
- **Severity breakdown:** {severity_breakdown}
"""
            
            # Determine issue type
            if len(found_keywords) > 0:
                highest_severity = max([sev for _, sev in found_keywords], 
                                     key=lambda x: ['LOW', 'MEDIUM', 'HIGH'].index(x))
                response += f"- **Severity Level:** {highest_severity}\n"
                response += f"- **Keywords Detected:** {', '.join([kw for kw, _ in found_keywords])}\n"
            
            if potential_app != "Unknown":
                response += f"- **Primary Application:** {potential_app}\n"
            
            response += f"""
### 🔥 Most Affected Applications
"""
            for app, count in app_dist.head(3).items():
                response += f"- **{app}:** {count} incidents\n"
            
            response += f"""
### ⚠️ Recent Alert Trend
"""
            for _, alert in recent_trend.head(3).iterrows():
                alert_date = pd.to_datetime(alert['date'], errors='coerce')
                if pd.isnull(alert_date):
                    time_ago = "Unknown"
                else:
                    time_ago = (datetime.now() - alert_date).total_seconds() / 3600
                    time_ago = f"{time_ago:.1f}h ago"
                response += f"- **{time_ago}:** {alert['subject'][:80]}... [{alert['severity']}]\n"
            
            # Generate specific recommendations
            response += f"""
### 🎯 Recommended Actions

"""
            
            if len(last_24h) > 5:
                response += "- 🚨 **High Activity Detected**: More than 5 similar incidents in last 24h\n"
            
            if 'HIGH' in severity_dist and severity_dist['HIGH'] > 0:
                response += "- 🔴 **Critical Priority**: High-severity incidents found in history\n"
                response += "- 📞 **Escalate**: Contact responsible team immediately\n"
            
            if len(app_dist) == 1:
                response += f"- 🎯 **Single Point of Failure**: All incidents from {app_dist.index[0]}\n"
            
            response += "- 📋 **Monitor**: Set up alerts for detected patterns\n"
            response += "- 🔍 **Investigate**: Check system logs for root cause\n"
            
            return {
                "response": response,
                "similar_emails": related_emails.drop(columns=['body']).to_dict('records'),
                "severity_analysis": severity_dist.to_dict(),
                "application_analysis": app_dist.to_dict(),
                "keywords_found": found_keywords,
                "mode": "enhanced_fast"
            }
        else:
            return {
                "response": f"""
## ⚡ Fast Analysis Results

**No related incidents found** in the last 7 days for: "{query}"

### 🎯 This Could Mean:
- ✅ **Stable System**: No recent similar issues
- 🆕 **New Problem**: First occurrence of this issue  
- 🔍 **Different Symptoms**: Try alternative keywords

### 🎯 Recommended Actions:
- Check if this is an ongoing issue
- Review system monitoring dashboards
- Use AI Analysis mode for deeper insights
- Contact relevant application teams
""",
                "similar_emails": [],
                "mode": "enhanced_fast"
            }
    except Exception as e:
        return {
            "response": f"Fast analysis failed: {str(e)}",
            "mode": "error"
        }


# Enhanced UI Components
def render_llm_search_interface():
    """Render the LLM search interface"""
    st.markdown("<h2 class='section-header'>🤖 AI-Powered Issue Analysis</h2>",
                unsafe_allow_html=True)

    # Initialize components
    if 'llm_config' not in st.session_state:
        st.session_state.llm_config = LLMConfig()

    if 'vector_db_manager' not in st.session_state:
        st.session_state.vector_db_manager = VectorDBManager()

    if 'llm_processor' not in st.session_state:
        st.session_state.llm_processor = LLMQueryProcessor(
            st.session_state.llm_config,
            st.session_state.vector_db_manager
        )

    # Analysis mode selection
    analysis_mode = st.radio(
        "Choose Analysis Mode:",
        ["⚡ Ultra Fast", "🚀 Fast Mode", "🤖 AI Analysis"],
        key="analysis_mode_selection"
    )

    # Search interface
    col1, col2 = st.columns([3, 1])

    with col1:
        user_query = st.text_area(
            "Describe the issue you're experiencing or ask a question about your system:",
            placeholder="e.g., 'Why is my database connection slow?' or 'What alerts are related to authentication failures?'",
            height=100,
            # Use the same UUID for consistency
            key="user_query_text"
        )

    with col2:
        st.write("")
        st.write("")
        search_button = st.button(
            "🔍 Analyze", use_container_width=True, key="search_button")

    if search_button and user_query:
        with st.spinner("Analyzing..."):
            ensure_qdrant_client_open()
            
            if analysis_mode == "⚡ Ultra Fast":
                result = enhanced_ultra_fast_analysis(user_query, st.session_state.vector_db_manager)
                if result.get("mode") == "error":
                    st.error(result["response"])
                else:
                    st.success("✅ Ultra Fast Analysis Complete")
                    st.markdown(result["response"])
                    # Show related emails
                    if result.get("similar_emails"):
                        with st.expander("📧 Related Historical Incidents", expanded=False):
                            for i, (doc, score) in enumerate(result["similar_emails"]):
                                st.markdown(f"**Incident {i+1}** (Similarity: {score:.3f})")
                                st.write(doc.page_content[:200] + "...")
                                if hasattr(doc, 'metadata'):
                                    st.json(doc.metadata)
                                st.markdown("---")
            elif analysis_mode == "🚀 Fast Mode":
                result = enhanced_fast_keyword_analysis(user_query)
                if result.get("mode") == "error":
                    st.error(result["response"])
                else:
                    st.success("✅ Fast Mode Analysis Complete")
                    st.markdown(result["response"])
                    # Show related emails
                    if result.get("similar_emails"):
                        with st.expander("📧 Related Historical Incidents", expanded=False):
                            df = pd.DataFrame(result["similar_emails"])
                            if not df.empty:
                                st.dataframe(df[['date', 'subject', 'severity', 'application']], use_container_width=True)
            elif analysis_mode == "🤖 AI Analysis":
                try:
                    with st.spinner("Performing AI analysis..."):
                        result = st.session_state.llm_processor.process_query(user_query)
                        if "error" in result:
                            st.error(f"Analysis failed: {result['error']}")
                        else:
                            st.success("✅ AI Analysis Complete")
                            st.markdown("### AI Analysis Results")
                            st.markdown(result["response"])
                            if result.get("context_emails"):
                                st.markdown("### Related Emails")
                                email_data = []
                                for i, (doc, score) in enumerate(result["context_emails"]):
                                    if hasattr(doc, 'metadata') and doc.metadata:
                                        email_data.append({
                                            "Relevance": f"{score:.2f}",
                                            "Date": doc.metadata.get('date', 'N/A'),
                                            "Sender": doc.metadata.get('sender', 'N/A'),
                                            "Subject": doc.metadata.get('subject', 'N/A'),
                                            "Severity": doc.metadata.get('severity', 'N/A'),
                                            "Application": doc.metadata.get('application', 'N/A')
                                        })
                                if email_data:
                                    email_df = pd.DataFrame(email_data)
                                    st.dataframe(email_df, use_container_width=True)
                                else:
                                    st.info("No closely related emails found in the database.")
                except Exception as e:
                    st.error(f"AI analysis failed with error: {str(e)}")
                    st.info("Try using Fast Mode instead, or check that your LLM model is properly configured.")
                    
def render_data_analytics():
    """Render enhanced data analytics with time frame filtering"""
    
    # Get selected time frame from session state
    selected_time_frame = st.session_state.get('time_frame_select', "Last Hour")
    selected_delta = time_frame_mapping.get(selected_time_frame, timedelta(hours=1))
    
    # Calculate the start date based on selected time frame
    start_date = (datetime.now(IST) - selected_delta).strftime('%Y-%m-%d %H:%M:%S')
    
    # Display current time frame selection
    st.info(f"📅 Showing data for: {selected_time_frame} (from {start_date})")
    
    # Initialize chart counter once at the beginning of the function
    chart_counter = 0
    
    try:
        conn = sqlite3.connect(DB_FILE)

        # Trend analysis using selected time frame
        st.markdown(f"### Trend Analysis ({selected_time_frame})")

        # Daily alert counts with dynamic time frame
        daily_alerts = pd.read_sql_query('''
            SELECT DATE(date) as day, COUNT(*) as alert_count, severity
            FROM emails
            WHERE date >= ?
            GROUP BY DATE(date), severity
            ORDER BY day
        ''', conn, params=(start_date,))

        # Top applications by alert count
        st.markdown(f"### Top Applications by Alert Count ({selected_time_frame})")
        app_alerts = pd.read_sql_query('''
            SELECT application, COUNT(*) as alert_count, severity
            FROM emails
            WHERE date >= ? AND application IS NOT NULL
            GROUP BY application, severity
            ORDER BY alert_count DESC
            LIMIT 10
        ''', conn, params=(start_date,))

        # Sender analysis
        st.markdown(f"### Alert Distribution by Sender ({selected_time_frame})")
        sender_alerts = pd.read_sql_query('''
            SELECT sender, COUNT(*) as alert_count, severity
            FROM emails
            WHERE date >= ?
            GROUP BY sender, severity
            ORDER BY alert_count DESC
            LIMIT 10
        ''', conn, params=(start_date,))

        # Get all email data for application criticality and severity analysis
        df = pd.read_sql_query('''
            SELECT date, sender, subject, severity, application as "App Name", keywords as "Critical Keywords"
            FROM emails
            WHERE date >= ?
            ORDER BY date DESC
        ''', conn, params=(start_date,))

        # Define resolution keywords
        RESOLUTION_KEYWORDS = [
            'resolved', 'fixed', 'completed', 'restored', 'back to normal', 'recovered']

        # Application Criticality
        st.markdown(
            f"<h2 class='section-header'>Application Criticality ({selected_time_frame})</h2>", unsafe_allow_html=True)
        try:
            keyword_data = []
            if not df.empty:
                df_filtered = df[~df["subject"].str.lower().apply(
                    lambda x: any(res_kw in x for res_kw in RESOLUTION_KEYWORDS))]
                for _, row in df_filtered.iterrows():
                    keywords = row["Critical Keywords"]
                    app_name = row["App Name"]
                    if keywords:
                        keyword_list = [kw.strip() for kw in keywords.split(',') if kw.strip()] if isinstance(
                            keywords, str) else (keywords if isinstance(keywords, list) else [])
                        for keyword in keyword_list:
                            keyword_data.append({
                                "Keyword": keyword,
                                "Application": app_name,
                                "Sender": row["sender"],
                                "Subject": row["subject"],
                                "Date": row["date"]
                            })

            if keyword_data:
                keyword_df = pd.DataFrame(keyword_data)
                keyword_counts = keyword_df.groupby(
                    ["Keyword", "Application"]).size().reset_index(name="Count")
                keyword_info = keyword_df.groupby(["Keyword", "Application"]).agg({
                    "Sender": lambda x: ", ".join(sorted(set(x))),
                    "Subject": lambda x: list(dict.fromkeys(x.head(5))),
                    "Date": lambda x: list(dict.fromkeys(x.head(5)))
                }).reset_index()
                keyword_info["Subject_With_Time"] = keyword_info.apply(
                    lambda row: "<br>".join(
                        [f"[{row['Date'][i]}] {row['Subject'][i]}" for i in range(
                            min(len(row['Subject']), 5))]
                    ) if isinstance(row['Subject'], list) and row['Subject'] and isinstance(row['Date'], list) and len(row['Date']) >= len(row['Subject'])
                    else "No subjects available",
                    axis=1
                )
                keyword_counts = keyword_counts.merge(
                    keyword_info[["Keyword", "Application",
                        "Sender", "Subject_With_Time"]],
                    on=["Keyword", "Application"],
                    how="left"
                )

                def get_severity(keyword):
                    for sev, kws in CRITICAL_KEYWORDS.items():
                        if keyword and isinstance(keyword, str) and keyword.lower() in [k.lower() for k in kws]:
                            return sev
                    return "NONE"
                keyword_counts["Severity"] = keyword_counts["Keyword"].apply(
                    get_severity)
                keyword_counts["Color"] = keyword_counts["Severity"].apply(
                lambda s: {"HIGH": "#ef4444", "MEDIUM": "#f59e42", "LOW": "#10b981"}.get(s, "#10b981"))

                # Create chart with application grouping
                fig_keyword_dist = px.bar(
                    keyword_counts,
                    x="Keyword",
                    y="Count",
                    color="Application",
                    height=600,
                    title=f"Keyword Distribution by Application ({selected_time_frame})",
                    custom_data=["Sender", "Subject_With_Time",
                        "Severity", "Application"]
                )
                fig_keyword_dist.update_traces(
                    hovertemplate=(
                        "<b>Keyword:</b> %{x}<br>"
                        "<b>Application:</b> %{customdata[3]}<br>"
                        "<b>Count:</b> %{y}<br>"
                        "<b>Severity:</b> %{customdata[2]}<br>"
                        "<b>Senders:</b> %{customdata[0]}<br>"
                        "<b>Subjects:</b><br>%{customdata[1]}<extra></extra>"
                    )
                )
                fig_keyword_dist.update_layout(
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="#ffffff",
                    font=dict(size=14, color="#1e293b"),
                    xaxis_title="Critical Keyword",
                    yaxis_title="Number of Emails",
                    xaxis=dict(tickangle=45, automargin=True,
                               tickfont=dict(size=12), gridcolor="#e2e8f0"),
                    yaxis=dict(gridcolor="#e2e8f0"),
                    bargap=0.2
                )
                # Better approach - use a consistent naming pattern with a counter
                chart_counter += 1
                st.plotly_chart(fig_keyword_dist, use_container_width=True,
                                key=f"chart_{chart_counter}")
            else:
                st.warning(
                    f"No critical keywords found in the selected time frame ({selected_time_frame}).")
        except Exception as e:
            st.error(f"Error in Critical Keyword Distribution: {e}")

        # Application Severity
        st.markdown(f"<h2 class='section-header'>Application Severity ({selected_time_frame})</h2>", unsafe_allow_html=True)

        try:
            if keyword_data:
                keyword_df = pd.DataFrame(keyword_data)
                keyword_counts = keyword_df.groupby(
                    ["Keyword", "Application"]).size().reset_index(name="Count")
                keyword_info = keyword_df.groupby(["Keyword", "Application"]).agg({
                    "Sender": lambda x: ", ".join(sorted(set(x))),
                    "Subject": lambda x: list(dict.fromkeys(x.head(5))),
                    "Date": lambda x: list(dict.fromkeys(x.head(5)))
                }).reset_index()
                keyword_info["Subject_With_Time"] = keyword_info.apply(
                    lambda row: "<br>".join(
                        [f"[{row['Date'][i]}] {row['Subject'][i]}" for i in range(
                            min(len(row['Subject']), 5))]
                    ) if isinstance(row['Subject'], list) and row['Subject'] and isinstance(row['Date'], list) and len(row['Date']) >= len(row['Subject'])
                    else "No subjects available",
                    axis=1
                )
                keyword_info["Subject_Copy_Text"] = keyword_info.apply(
                    lambda row: "\n".join(
                        [f"{row['Subject'][i]}" for i in range(
                            min(len(row['Subject']), 5))]
                    ) if isinstance(row['Subject'], list) and row['Subject']
                    else "No subjects available",
                    axis=1
                )
                keyword_counts = keyword_counts.merge(
                    keyword_info[["Keyword", "Application", "Sender",
                        "Subject_With_Time", "Subject_Copy_Text"]],
                    on=["Keyword", "Application"],
                    how="left"
                )

                def get_severity(keyword):
                    for sev, kws in CRITICAL_KEYWORDS.items():
                        if keyword and isinstance(keyword, str) and keyword.lower() in [k.lower() for k in kws]:
                            return sev
                    return "NONE"
                keyword_counts["Severity"] = keyword_counts["Keyword"].apply(
                    get_severity)
                keyword_counts["Color"] = keyword_counts["Severity"].apply(
                    lambda s: {"HIGH": "#ef4444", "MEDIUM": "#f59e42", "LOW": "#10b981"}.get(s, "#10b981"))

                # Create application with severity for x-axis display
                keyword_counts["Application_With_Severity"] = keyword_counts.apply(
                    lambda row: f"{row['Application']} [{row['Severity']}]" if row['Severity'] != "NONE" else row['Application'],
                    axis=1
                )

                # Create chart with application and severity grouping
                fig_keyword_dist = px.bar(
                    keyword_counts,
                    x="Application_With_Severity",
                    y="Count",
                    color="Keyword",
                    title=f"Distribution of Applications by Severity ({selected_time_frame})",
                    height=800,
                    custom_data=["Sender", "Subject_With_Time", "Severity",
                        "Application", "Keyword", "Subject_Copy_Text"]
                )
                fig_keyword_dist.update_traces(
                    hovertemplate=(
                        "<b>Application:</b> %{x}<br>"
                        "<b>Keyword:</b> %{customdata[4]}<br>"
                        "<b>Count:</b> %{y}<br>"
                        "<b>Severity:</b> %{customdata[2]}<br>"
                        "<b>Senders:</b> %{customdata[0]}<br>"
                        "<b>Subjects:</b><br>%{customdata[1]}<extra></extra>"
                    )
                )
                fig_keyword_dist.update_layout(
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="#ffffff",
                    font=dict(size=14, color="#1e293b"),
                    xaxis_title="Application",
                    yaxis_title="Number of Emails",
                    xaxis=dict(tickangle=45, automargin=True,
                               tickfont=dict(size=12), gridcolor="#e2e8f0"),
                    yaxis=dict(gridcolor="#e2e8f0"),
                    bargap=0.2
                )
                # Better approach - use a consistent naming pattern with a counter
                chart_counter += 1
                st.plotly_chart(fig_keyword_dist, use_container_width=True,
                                key=f"chart_{chart_counter}")

                # Add copy interface for Application Severity
                if not keyword_counts.empty:
                    st.markdown("---")
                    st.markdown(
                        f"<h4 style='color: #1e293b;'>📋 Subject List ({selected_time_frame})</h4>", unsafe_allow_html=True)

                    # Create a selectbox for applications
                    applications = keyword_counts['Application'].unique().tolist()
                    selected_app = st.selectbox(
                        "Select Application:", applications, key="app_selection")
                    
                    # Complete the functionality for selected application
                    if selected_app:
                        app_data = keyword_counts[keyword_counts['Application'] == selected_app]
                        app_subjects = []

                        for _, row in app_data.iterrows():
                            if pd.notna(row.get('Subject_Copy_Text')):
                                app_subjects.append(row['Subject_Copy_Text'])
                        
                        if app_subjects:
                            st.text_area("Subjects:", value="\n\n".join(app_subjects), 
                                        height=300, key="subject_copy_area")
            else:
                st.warning(f"No critical keywords found in the selected time frame ({selected_time_frame}).")
        except Exception as e:
            st.error(f"Error processing keyword data: {e}")

        # Add trend visualization for selected time frame
        st.markdown(f"<h2 class='section-header'>Alert Trend ({selected_time_frame})</h2>", unsafe_allow_html=True)
        
        try:
            # Get alert trend data with hourly or daily grouping based on time frame
            if selected_delta < timedelta(days=1):
                # For shorter time frames, group by hour
                trend_data = pd.read_sql_query('''
                    SELECT 
                        strftime('%Y-%m-%d %H:00:00', date) as time_period,
                        COUNT(*) as alert_count,
                        severity
                    FROM emails
                    WHERE date >= ?
                    GROUP BY time_period, severity
                    ORDER BY time_period
                ''', conn, params=(start_date,))
                trend_data['time_period'] = pd.to_datetime(trend_data['time_period'])
                x_title = "Hour"
            else:
                # For longer time frames, group by day
                trend_data = pd.read_sql_query('''
                    SELECT 
                        DATE(date) as time_period,
                        COUNT(*) as alert_count,
                        severity
                    FROM emails
                    WHERE date >= ?
                    GROUP BY time_period, severity
                    ORDER BY time_period
                ''', conn, params=(start_date,))
                trend_data['time_period'] = pd.to_datetime(trend_data['time_period'])
                x_title = "Date"
            
            if not trend_data.empty:
                # Create trend chart
                fig_trend = px.line(
                    trend_data,
                    x="time_period",
                    y="alert_count",
                    color="severity",
                    title=f"Alert Trend ({selected_time_frame})",
                    markers=True,
                    height=500
                )
                fig_trend.update_layout(
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="#ffffff",
                    font=dict(size=14, color="#1e293b"),
                    xaxis_title=x_title,
                    yaxis_title="Number of Alerts",
                    xaxis=dict(gridcolor="#e2e8f0"),
                    yaxis=dict(gridcolor="#e2e8f0"),
                    legend_title="Severity"
                )
                chart_counter += 1
                st.plotly_chart(fig_trend, use_container_width=True, key=f"chart_{chart_counter}")
            else:
                st.warning(f"No alert trend data available for the selected time frame ({selected_time_frame}).")
                
        except Exception as e:
            st.error(f"Error creating trend visualization: {e}")
    
    except Exception as e:
        st.error(f"Error loading analytics data: {e}")
              
# After the time frame selection in the sidebar
time_frame_mapping = {
    "Last 5 Minutes": timedelta(minutes=5),
    "Last 10 Minutes": timedelta(minutes=10),
    "Last 15 Minutes": timedelta(minutes=15),
    "Last 30 Minutes": timedelta(minutes=30),
    "Last Hour": timedelta(hours=1),
    "Last 4 Hours": timedelta(hours=4),
    "Last 12 Hours": timedelta(hours=12),
    "Last 24 Hours": timedelta(days=1),
    "Last 3 Days": timedelta(days=3),
    "Last Week": timedelta(days=7)
}

# Initialize the selected time frame in session state
if 'selected_time_frame' not in st.session_state:
    st.session_state.selected_time_frame = "Last Hour"

# These variables will be set later in the sidebar
start_date = None
end_date = None


def get_oauth_token():
    """Get OAuth token for Microsoft Graph API"""
    try:
        # Get token from Microsoft identity platform
        token_url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
        token_data = {
            'grant_type': 'client_credentials',
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
            'scope': SCOPE
        }
        
        token_response = requests.post(token_url, data=token_data)
        token_response.raise_for_status()  # Raise exception for HTTP errors
        
        # Extract access token
        access_token = token_response.json().get('access_token')
        if not access_token:
            logging.error("No access token received in response")
            return None
            
        return access_token
    except Exception as e:
        logging.error(f"Failed to get OAuth token: {e}")
        return None
    
# Add auto-refresh toggle and interval selection
st.sidebar.markdown("---")
st.sidebar.subheader("Auto-Refresh Settings")
auto_refresh_enabled = st.sidebar.checkbox("Enable Auto-Refresh", value=True, key="auto_refresh_enabled")

if auto_refresh_enabled:
    refresh_interval = st.sidebar.slider(
        "Refresh Interval (seconds)",
        min_value=5,
        max_value=900,
        value=60,
        step=5,
        key="refresh_interval"
    )
    
    # Display time until next refresh
    if 'last_refresh_time' in st.session_state:
        time_since_refresh = (datetime.now(IST) - st.session_state.last_refresh_time).total_seconds()
        time_until_refresh = max(0, refresh_interval - time_since_refresh)
        st.sidebar.progress(1 - (time_until_refresh / refresh_interval))
        st.sidebar.text(f"Next refresh in: {int(time_until_refresh)} seconds")

# Inside main(), replace the current auto-refresh section with this:
if 'last_refresh_time' not in st.session_state:
    st.session_state.last_refresh_time = datetime.now(IST)

# Handle auto-refresh based on user settings
if st.session_state.get('auto_refresh_enabled', True):
    refresh_interval = st.session_state.get('refresh_interval', 60)
    
    # Set up auto-refresh
    st_autorefresh(interval=refresh_interval * 1000, key=f"autorefresh_{uuid.uuid4()}")
    
    # Auto-fetch emails if enough time has passed
    time_since_refresh = (datetime.now(IST) - st.session_state.last_refresh_time).total_seconds()
    if time_since_refresh >= refresh_interval:
        # Update last refresh time
        st.session_state.last_refresh_time = datetime.now(IST)
        
        # Fetch emails based on selected time frame
        try:
            # Get access token
            access_token = get_oauth_token()
            if access_token:
                # Calculate date range based on selected time frame
                selected_delta = time_frame_mapping[st.session_state.selected_time_frame]
                start_date = datetime.now(IST) - selected_delta
                end_date = datetime.now(IST)
                
                # Fetch and process emails
                df, messages = fetch_emails(start_date, end_date, access_token)
                if not df.empty:
                    process_and_store_emails(df)
                    st.session_state.auto_fetch_msg = f"Auto-fetched and stored {len(df)} emails based on {st.session_state.selected_time_frame} time frame"
                    logging.info(st.session_state.auto_fetch_msg)
        except Exception as e:
            logging.error(f"Auto-fetch error: {str(e)}")
            st.error(f"Auto-fetch error: {str(e)}")


# Add session cleanup in main function
def main():

    """Main function to render the complete dashboard"""
    # Apply custom CSS for better styling
    st.markdown("""
    <style>
    .section-header {color: #1e88e5; border-bottom: 1px solid #e0e0e0; padding-bottom: 10px;}
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar components
    with st.sidebar:
        st.title("📊 Email Alert Dashboard")
        st.markdown("---")
        
        # Time frame selection - make it more prominent for analytics
        st.markdown("### 📆 Time Frame Selection")
        st.markdown("_This affects both data fetching and analytics views_")
        
        time_frame = st.selectbox(
            "Select Time Frame:",
            ["Last 5 Minutes", "Last 10 Minutes", "Last 15 Minutes", "Last 30 Minutes", 
            "Last Hour", "Last 4 Hours", "Last 12 Hours", "Last 24 Hours", "Last 3 Days", "Last Week"],
            key="time_frame_select"
        )
        
        # Store selection in session state
        st.session_state.selected_time_frame = time_frame
        
        # Show time range for clarity
        selected_delta = time_frame_mapping[time_frame]
        start_date = (datetime.now(IST) - selected_delta).strftime('%Y-%m-%d %H:%M:%S')
        end_date = datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
        st.info(f"From: {start_date}\nTo: {end_date}")
        
        # Manual fetch button
        if st.sidebar.button("Fetch New Emails Now", key="fetch_emails_button"):
            with st.spinner("Fetching emails..."):
                try:
                    # Get access token
                    access_token = get_oauth_token()
                    if access_token:
                        # Calculate date range based on selected time frame
                        selected_delta = time_frame_mapping[time_frame]
                        start_date = datetime.now(IST) - selected_delta
                        end_date = datetime.now(IST)
                        
                        # Fetch and process emails
                        df, messages = fetch_emails(start_date, end_date, access_token)
                        if not df.empty:
                            process_and_store_emails(df)
                            st.success(f"Successfully fetched and stored {len(df)} emails!")
                            st.session_state.last_refresh_time = datetime.now(IST)
                        else:
                            st.info("No new emails found in the selected time frame.")
                    else:
                        st.error("Failed to get OAuth token")
                except Exception as e:
                    st.error(f"Error fetching emails: {str(e)}")
                    logging.error(f"Manual fetch error: {str(e)}")
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📧 Recent Emails", "🤖 AI Analysis", "📊 Analytics", "⚙️ Settings"])
    
    with tab1:
        render_email_list()
    
    with tab2:
        render_llm_search_interface()
    
    with tab3:
        render_data_analytics()
    
    with tab4:
        render_settings()
    
    # Handle Qdrant client cleanup when done
    if st.session_state.get('vector_db_manager') and hasattr(st.session_state.vector_db_manager, 'client'):
        try:
            st.session_state.vector_db_manager.client.close()
        except:
            pass

    # Refresh dashboard automatically based on severity
    if "high_priority_emails" in st.session_state and st.session_state.high_priority_emails > 0:
        refresh_seconds = NOTIFICATION_CONFIG['high_priority_refresh_seconds']
    elif "medium_priority_emails" in st.session_state and st.session_state.medium_priority_emails > 0:
        refresh_seconds = NOTIFICATION_CONFIG['medium_priority_refresh_seconds']
    else:
        refresh_seconds = NOTIFICATION_CONFIG['low_priority_refresh_seconds']

    # Auto-refresh the dashboard
    if 'last_refresh_time' not in st.session_state:
        st.session_state.last_refresh_time = datetime.now(IST)

    # Handle auto-refresh based on user settings
    if st.session_state.get('auto_refresh_enabled', True):
        refresh_interval = st.session_state.get('refresh_interval', 60)
        
        # Set up auto-refresh
        st_autorefresh(interval=refresh_interval * 1000, key=f"autorefresh_{uuid.uuid4()}")
        
        
        # Auto-fetch emails if enough time has passed
        time_since_refresh = (datetime.now(IST) - st.session_state.last_refresh_time).total_seconds()
        if time_since_refresh >= refresh_interval:
            # Update last refresh time
            st.session_state.last_refresh_time = datetime.now(IST)
            
            # Fetch emails based on selected time frame
            try:
                # Get access token
                access_token = get_oauth_token()
                if access_token:
                  # Calculate date range based on selected time frame
                    selected_delta = time_frame_mapping[st.session_state.selected_time_frame]
                    start_date = datetime.now(IST) - selected_delta
                    end_date = datetime.now(IST)
                    
                    # Fetch and process emails
                    df, messages = fetch_emails(start_date, end_date, access_token)
                    if not df.empty:
                        process_and_store_emails(df)
                        st.session_state.auto_fetch_msg = f"Auto-fetched and stored {len(df)} emails based on {st.session_state.selected_time_frame} time frame"
                        logging.info(st.session_state.auto_fetch_msg)
            except Exception as e:
                logging.error(f"Auto-fetch error: {str(e)}")
                st.error(f"Auto-fetch error: {str(e)}")


# Call the main function to render the dashboard
if __name__ == "__main__":
    init_enhanced_db()  # Make sure DB is initialized
    main()
