import streamlit as st
import psycopg2
from psycopg2 import extras
import google.generativeai as genai
import numpy as np
import textwrap
import datetime
import os
import tempfile
import time
import subprocess
import sys
import json # Need json to dump the SA key dict
import atexit # Import atexit for cleanup

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Agent Context Summarizer (DB RAG)", layout="wide")

# Import Vertex AI libraries for embedding
import vertexai
from vertexai.language_models import TextEmbeddingModel

# Import pgvector for psycopg2 type handling
from pgvector.psycopg2 import register_vector

# --- Configuration ---
# Load the SA JSON from Streamlit secrets
# Access secrets from .streamlit/secrets.toml
GOOGLE_CLOUD_PROJECT = st.secrets["GOOGLE_CLOUD_PROJECT"]
GOOGLE_CLOUD_LOCATION = st.secrets["GOOGLE_CLOUD_LOCATION"]
# The proxy will connect to 127.0.0.1 using PG_PORT
# The application connects to the proxy at 127.0.0.1:PG_PORT
# So PG_HOST should effectively be '127.0.0.1' for the application connection
PG_HOST_APP = '127.0.0.1' # Application connects to localhost proxy
PG_PORT = st.secrets["postgres"]["port"] # Local port the proxy listens on (e.g., 5432)

# Database connection details used by the *application* connecting to the proxy
PG_DATABASE = st.secrets["postgres"]["database"]
PG_USER = st.secrets["postgres"]["user"]
# PG_PASSWORD is NOT needed by the application when using IAM authentication via proxy
# If your DB user requires a password *even with IAM auth*, you might need it,
# but typically you configure the Cloud SQL user to use IAM.
# PG_PASSWORD = st.secrets["postgres"]["password"]

# Cloud SQL Instance Connection Name (used by the proxy)
PG_INSTANCE_CONNECTION_NAME = st.secrets["postgres"]["instance_connection_name"]


# LLM Configuration (for chat response)
LLM_MODEL_NAME = 'gemini-2.0-flash'
GENAI_API_KEY = st.secrets["GEMINI_API_KEY"] # Assume Gemini API Key is in secrets

# Embedding Model Configuration (using Vertex AI)
EMBEDDING_MODEL_NAME = "text-embedding-005"
EMBEDDING_DIMENSION = 768
RETRIEVAL_LIMIT = 50

# --- Cloud SQL Auth Proxy Setup ---
# Use session state to manage the proxy process and temp file path across reruns
if 'cloudsql_proxy_process' not in st.session_state:
    st.session_state['cloudsql_proxy_process'] = None
if 'cloudsql_temp_key_path' not in st.session_state:
     st.session_state['cloudsql_temp_key_path'] = None

def cleanup_cloudsql_proxy():
    """Terminates the Cloud SQL Auth Proxy process and cleans up temp key file."""
    print("Running Cloud SQL Proxy cleanup...")
    # Terminate the proxy process if it's running
    if st.session_state.get('cloudsql_proxy_process') is not None:
        process = st.session_state['cloudsql_proxy_process']
        if process.poll() is None: # Check if process is still running
            print(f"Terminating Cloud SQL Proxy process (PID: {process.pid})...")
            try:
                process.terminate()
                # Give it a moment to exit gracefully
                process.wait(timeout=5)
                print("Cloud SQL Proxy process terminated.")
            except subprocess.TimeoutExpired:
                 print("Cloud SQL Proxy did not terminate gracefully, killing it.")
                 process.kill()
            except Exception as e:
                print(f"Error terminating proxy process: {e}")
        st.session_state['cloudsql_proxy_process'] = None # Reset state

    # Clean up the temporary service account key file
    temp_key_path = st.session_state.get('cloudsql_temp_key_path')
    if temp_key_path and os.path.exists(temp_key_path):
        print(f"Deleting temporary service account key file: {temp_key_path}")
        try:
            os.unlink(temp_key_path)
            print("Temporary key file deleted.")
        except Exception as e:
            print(f"Error deleting temporary key file {temp_key_path}: {e}")
        st.session_state['cloudsql_temp_key_path'] = None # Reset state
    print("Cloud SQL Proxy cleanup finished.")

# Register the cleanup function to run when the script exits
# Note: atexit is not guaranteed to run in all environments/exit scenarios,
# but it's a good practice for basic cleanup.
atexit.register(cleanup_cloudsql_proxy)


def start_cloudsql_proxy():
    """Starts the Cloud SQL Auth Proxy as a subprocess."""
    # Check if proxy is already running based on session state
    if st.session_state.get('cloudsql_proxy_process') is not None:
        # Check if the process is actually still alive
        if st.session_state['cloudsql_proxy_process'].poll() is None:
             print("Cloud SQL Auth Proxy is already running and appears healthy.")
             return # Proxy is running, do nothing
        else:
             # Process terminated unexpectedly, clean up state
             print("Cloud SQL Auth Proxy process found in state but is not running. Cleaning up state.")
             cleanup_cloudsql_proxy() # Clean up old state before restarting

    print("Attempting to start Cloud SQL Auth Proxy...")

    # --- 1. Get Service Account key from secrets and save to a temporary file ---
    try:
        # We need the dictionary representation of the secret for json.dump
        sa_key_dict = dict(st.secrets["gcp"])

        # Create a temporary file to store the key securely
        # delete=False means we are responsible for deleting it
        # Use mkstemp for better security than NamedTemporaryFile(delete=False)
        fd, temp_key_file_path = tempfile.mkstemp(suffix=".json", text=True)
        with os.fdopen(fd, 'w') as tmp:
             json.dump(sa_key_dict, tmp)

        st.session_state['cloudsql_temp_key_path'] = temp_key_file_path
        print(f"Service account key saved to temporary file: {temp_key_file_path}")

    except Exception as e:
        st.error(f"Failed to save service account key to temporary file for Cloud SQL Proxy: {e}")
        # Cleanup already handled by atexit or will be handled before retry
        return


    # --- 2. Define the proxy command ---
    # Assuming you put the executable in the root dir or it's in the PATH
    proxy_executable = "./cloud-sql-proxy"

    # Check if the executable exists and is runnable
    if not os.path.exists(proxy_executable):
         st.error(f"Cloud SQL Auth Proxy executable not found at {proxy_executable}. Make sure 'cloud-sql-proxy' file is in your repo root and is executable (chmod +x).")
         cleanup_cloudsql_proxy() # Clean up the temp file
         return
    if not os.access(proxy_executable, os.X_OK):
         st.error(f"Cloud SQL Auth Proxy executable at {proxy_executable} is not executable. Run 'chmod +x ./cloud-sql-proxy' in your repository.")
         cleanup_cloudsql_proxy() # Clean up the temp file
         return


    # Construct the command using the correct syntax (positional instance arg)
    # Use --auto-iam-authn for IAM login
    command = [
        proxy_executable,
        f"{PG_INSTANCE_CONNECTION_NAME}=tcp:{PG_HOST_APP}:{PG_PORT}",
        "--auto-iam-authn", # Use the correct flag for IAM authentication
        # "-verbose", # Optional: Uncomment for more proxy logging
        # Use --credentials-file to point to the temporary key file
        f"--credentials-file={st.session_state['cloudsql_temp_key_path']}"
    ]
    print(f"Cloud SQL Auth Proxy command: {' '.join(command)}") # Log the command being run

    # --- 3. Start the subprocess ---
    # Use subprocess.Popen to run in the background
    # Capture stdout/stderr to check for immediate errors
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True # Decode stdout/stderr as text
            # Consider setting a working directory if needed: cwd="."
        )
        st.session_state['cloudsql_proxy_process'] = process
        print(f"Cloud SQL Auth Proxy subprocess started with PID: {process.pid}")

        # Give the proxy a moment to start up and check for immediate failure
        # A short sleep + poll check is more robust than just sleep
        startup_check_time = 5 # seconds
        start_time = time.time()
        while (time.time() - start_time) < startup_check_time:
             if process.poll() is not None: # Check if process has exited
                  # Proxy failed to start, read stderr for errors
                  stdout, stderr = process.communicate() # Read remaining output
                  st.error(f"Cloud SQL Auth Proxy failed to start or terminated unexpectedly (Return code: {process.returncode}).")
                  st.text(f"Proxy STDOUT:\n{stdout}")
                  st.text(f"Proxy STDERR:\n{stderr}")
                  cleanup_cloudsql_proxy() # Clean up the temp file and reset state
                  return # Exit the startup function

             time.sleep(0.1) # Small sleep before checking again

        # If we reached here, the process is likely still running after the check time
        if process.poll() is None:
             print("Cloud SQL Auth Proxy seems to be running after initial check.")
             # You might want to read the first few lines of stdout/stderr here too
             # to see startup messages, but be careful not to block.
             # Example: first_line = process.stderr.readline()
             pass # Proxy started successfully
        else:
             # This case should ideally be caught by the loop above, but as a fallback
             print("Cloud SQL Auth Proxy process check after sleep found it exited.")
             stdout, stderr = process.communicate()
             st.error(f"Cloud SQL Auth Proxy failed check after startup time (Return code: {process.returncode}).")
             st.text(f"Proxy STDOUT:\n{stdout}")
             st.text(f"Proxy STDERR:\n{stderr}")
             cleanup_cloudsql_proxy() # Clean up
             return


    except Exception as e:
        st.error(f"An error occurred trying to start the Cloud SQL Auth Proxy subprocess: {e}")
        cleanup_cloudsql_proxy() # Clean up on failure
        return

    # Add a small delay after successful apparent start before attempting DB connection
    # to give the proxy time to open the port
    time.sleep(2) # Adjust if needed

# --- Call the function to start the proxy early in your script ---
# This ensures it runs when the app container starts or reruns without a cached proxy
# Use a button or initial load check to trigger it
if st.session_state['cloudsql_proxy_process'] is None:
     # You could add a button here, or just auto-start on first load attempt
     # if st.button("Start Cloud SQL Proxy"):
     start_cloudsql_proxy()
     # Optional: st.rerun() could be used after starting the proxy
     # if you want the script to restart and potentially connect immediately.
     # However, caching functions help avoid needing a full rerun.


# --- Initialize Vertex AI and Embedding Model (cached) ---
@st.cache_resource
def initialize_vertex_ai_and_embedding_model():
    """Initializes Vertex AI and loads the embedding model."""
    try:
        # Use credentials object directly if loaded via google.oauth2
        vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION, credentials=st.secrets.get("gcp_credentials_object"))
        print(f"Vertex AI initialized for project '{GOOGLE_CLOUD_PROJECT}' in location '{GOOGLE_CLOUD_LOCATION}'.")
        embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
        print(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded.")
        return embedding_model
    except Exception as e:
        st.error(f"Error initializing Vertex AI or loading embedding model: {e}")
        # Provide more context on potential causes
        st.info("Please ensure:")
        st.info("- `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` are correct in secrets.")
        st.info("- Your service account key JSON in `secrets.toml` is valid under `[gcp]`.")
        st.info("- The service account has the `Vertex AI User` role.")
        st.stop()

# Load the credentials object directly from secrets for Vertex AI initialization
# Assuming your secrets structure saves the loaded credentials object directly if possible
# If not, load it here from the dict: service_account.Credentials.from_service_account_info(st.secrets["gcp"])
# Let's adjust the vertexai.init call to use the loaded object:
try:
    st.secrets["gcp_credentials_object"] = service_account.Credentials.from_service_account_info(st.secrets["gcp"])
except Exception as e:
     st.error(f"Failed to load GCP service account credentials from secrets: {e}")
     st.stop()


embedding_model = initialize_vertex_ai_and_embedding_model()


# --- Initialize LLM (cached) ---
@st.cache_resource
def get_generative_model():
     """Initializes and returns the generative model for chat."""
     try:
        genai.configure(api_key=GENAI_API_KEY)
        # Check if API key is empty
        if not GENAI_API_KEY:
             st.warning("GEMINI_API_KEY is not set in Streamlit secrets.")
             return None
        return genai.GenerativeModel(LLM_MODEL_NAME)
     except Exception as e:
         st.error(f"Error configuring Gemini API for chat: {e}")
         st.stop()

chat_model = get_generative_model()
# Add a check here if the chat model failed to initialize
if chat_model is None:
    st.error("Chat model initialization failed. Please check API key and configuration.")
    # Decide if you want to stop the app here
    # st.stop()


# --- Database Functions ---

# Use st.cache_resource to cache the connection to reuse it across interactions
# Hash the important connection parameters so a change triggers recaching
@st.cache_resource(hash_funcs={
    type(st.session_state): lambda _: None, # Don't hash the whole session state
    subprocess.Popen: lambda _: None # Don't hash the process object
    }, show_spinner="Connecting to database...")
def get_db_connection(conn_params):
    """Establishes and returns a database connection."""
    # Ensure proxy is running before attempting connection
    # This check happens *within* the cached function the first time it's called
    # or if params change. Subsequent calls within the same session state rerun
    # hit the cache, but the proxy status should be checked before *using* the connection.
    if st.session_state.get('cloudsql_proxy_process') is None or \
       st.session_state['cloudsql_proxy_process'].poll() is not None:
         st.error("Cloud SQL Auth Proxy is not running or has terminated. Cannot connect to database.")
         # You could attempt to restart it here, but be careful with infinite loops
         # start_cloudsql_proxy()
         return None # Indicate failure

    try:
        print(f"Attempting database connection to {conn_params['host']}:{conn_params['port']}...")
        conn = psycopg2.connect(**conn_params)
        # Register the vector type with psycopg2 using pgvector library
        register_vector(conn)
        print("pgvector.psycopg2.register_vector called.")
        print("Database connection successful!")
        return conn
    except Exception as e:
        st.error(f"Error connecting to database (check proxy status and connection details): {e}")
        # Attempt to read proxy logs if connection fails (might not get much here immediately)
        # print("\n--- Attempting to read recent proxy stderr ---")
        # try:
        #     proxy_stderr = st.session_state['cloudsql_proxy_process'].stderr
        #     if proxy_stderr:
        #          # This might only get output buffered *up to this point*
        #          # Reading too much might consume output meant for the console
        #          recent_output = proxy_stderr.read()
        #          print(recent_output)
        # except Exception as log_e:
        #     print(f"Error reading proxy stderr: {log_e}")
        return None

# Prepare connection parameters dictionary to pass to the cached function
# This includes only the parameters necessary for psycopg2.connect
db_connection_params = {
    "host": PG_HOST_APP, # Connect to the proxy's local address
    "database": PG_DATABASE,
    "user": PG_USER,
    "port": PG_PORT, # Connect to the proxy's local port
    # "password": PG_PASSWORD # Omit password when using IAM auth via proxy
}

conn = get_db_connection(db_connection_params)

# Add a check if the database connection failed
if conn is None:
    st.error("Database connection failed. Please check proxy and database configuration.")
    # Decide if you want to stop the app here or allow loading static parts
    # st.stop()


# Commented out the log_message function as requested
# def log_message(company_id, role, content): ...

# --- Embedding Function (Synchronous for app.py) ---
def generate_embedding_sync(text):
    """Generates an embedding for a single text using the Vertex AI model synchronously."""
    # Add a check if embedding model failed to load
    if embedding_model is None:
        st.error("Embedding model is not loaded. Cannot generate embeddings.")
        return None

    if not text or not str(text).strip():
         return np.zeros(EMBEDDING_DIMENSION).tolist() # Return zero vector for empty/whitespace

    try:
        embeddings = embedding_model.get_embeddings([str(text)])
        return embeddings[0].values
    except Exception as e:
        st.error(f"Error generating embedding for text: '{str(text)[:50]}...' - {e}")
        # Provide more context
        st.info("Please ensure:")
        st.info("- `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` are correct.")
        st.info("- The service account key JSON is valid and has `Vertex AI User` role.")
        st.info("- The text-embedding-005 model is available in your region.")
        return None


# --- Context Retrieval and Formatting Functions ---

def get_general_company_context(company_id):
    """Retrieves general context for a given company_id from structured tables."""
    conn = get_db_connection(db_connection_params) # Get connection using the cached function
    if conn is None:
        return None # Cannot get context without connection

    context_data = {}
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # ... (rest of your context retrieval logic) ...
            # 1. Company Information (using 'id' as PK)
            cur.execute('SELECT * FROM companies WHERE id = %s', (company_id,))
            company_info = cur.fetchone()
            if company_info:
                 context_data['company_info'] = dict(company_info)
            else:
                 st.warning(f"No company found for ID: {company_id}")
                 return None # Exit if the company doesn't exist

            # 2. Financial Years (using 'id' as PK and 'companyId' FK)
            cur.execute('SELECT * FROM "fiscalYears" WHERE "companyId" = %s ORDER BY "endDate" DESC', (company_id,))
            context_data['fiscal_years'] = cur.fetchall()

            # Add other structured data retrieval here if needed (e.g., key contacts)

        return context_data

    except Exception as e:
        st.error(f"Error retrieving general company context from database: {e}")
        return None

# ... (format_general_company_context_for_llm remains the same) ...
def format_general_company_context_for_llm(context_data):
    """Formats the general retrieved database context into a string for the LLM prompt."""
    if not context_data:
        return "No general context information available for this company."

    formatted_string = "--- General Company Context ---\n\n"

    # Company Info
    if 'company_info' in context_data and context_data['company_info']:
        info = context_data['company_info']
        formatted_string += f"Company ID: {info.get('id', 'N/A')}\n"
        formatted_string += f"Company Name: {info.get('name', 'N/A')}\n"
        formatted_string += f"Status: {info.get('status', 'N/A')}\n"
        formatted_string += f"Type: {info.get('type', 'N/A')}\n"
        formatted_string += f"Functional Currency: {info.get('functionalCurrency', 'N/A')}\n"
        formatted_string += f"Tags: {info.get('tags', 'N/A')}\n"
        formatted_string += f"Opening Balance Date: {info.get('openingBalanceDate', 'N/A')}\n"
        formatted_string += f"Soft Lock Date: {info.get('softLockDate', 'N/A')}\n"
        formatted_string += f"Hard Lock Date: {info.get('hardLockDate', 'N/A')}\n"
        # Add other relevant company fields from your schema
        formatted_string += "\n"

    # Financial Years
    if 'fiscal_years' in context_data and context_data['fiscal_years']:
        formatted_string += "Fiscal Years:\n"
        # Ensure the dictionary values are accessed correctly (e.g., .get() or ['key'])
        # and handle potential non-string types for printing
        for fy in context_data['fiscal_years']:
             # Adjust formatting based on your schema (e.g., id, endDate, rawData)
            formatted_string += f"- ID: {fy.get('id', 'N/A')}, End Date: {fy.get('endDate', 'N/A')}, Raw Data Snippet: {str(fy.get('rawData', 'N/A'))[:100]}...\n" # Truncate raw data
        formatted_string += "\n"

    formatted_string += "--- End General Company Context ---\n\n"

    return formatted_string


def retrieve_relevant_snippets_rag(company_id, query_embedding, limit=RETRIEVAL_LIMIT):
    conn = get_db_connection(db_connection_params) # Get connection using the cached function
    if conn is None:
        return [] # Cannot retrieve snippets without connection

    # Add check if query embedding failed
    if query_embedding is None:
        st.error("Query embedding is missing. Cannot perform RAG search.")
        return []

    retrieved_snippets = []

    # Adjust company_id variable usage based on your schema details
    # Assuming past_messages uses ID without comma, notes/tickets use ID with comma
    company_id_for_past_messages = company_id.replace(',', '') if isinstance(company_id, str) else str(company_id)
    company_id_for_notes_tickets = company_id # Use the original value

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # --- Retrieve from past_messages ---
            print(f"Querying past_messages for companyId (no comma): '{company_id_for_past_messages}'")
            # Ensure the column name 'message_from_who' is correct if it exists
            cur.execute(
                """
                SELECT id, "createdAt", 'unknown' as message_from_who, text -- Using 'unknown' if message_from_who column doesn't exist
                FROM past_messages
                WHERE "companyId" = %s AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (company_id_for_past_messages, query_embedding, limit)
            )
            messages_snippets = cur.fetchall()
            print(f"Retrieved {len(messages_snippets)} past messages for company {company_id}")
            for snippet in messages_snippets:
                 retrieved_snippets.append({
                     'source': 'past_message',
                     'id': snippet['id'],
                     'createdAt': snippet['createdAt'],
                     'sender': snippet.get('message_from_who', 'unknown'), # Use .get for safety
                     'content': snippet['text']
                 })

            # --- Retrieve from notes ---
            print(f"Querying notes for companyId (with comma): '{company_id_for_notes_tickets}'")
            cur.execute(
                """
                SELECT id, "createdAt", "lastModifiedByUserId", text
                FROM notes
                WHERE "companyId" = %s AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (company_id_for_notes_tickets, query_embedding, limit)
            )
            notes_snippets = cur.fetchall()
            print(f"Retrieved {len(notes_snippets)} notes for company {company_id}")
            for snippet in notes_snippets:
                 retrieved_snippets.append({
                     'source': 'note',
                     'id': snippet['id'],
                     'createdAt': snippet['createdAt'],
                     'author': snippet['lastModifiedByUserId'],
                     'content': snippet['text']
                 })

            # --- Retrieve from tickets ---
            print(f"Querying tickets for companyId (with comma): '{company_id_for_notes_tickets}'")
            # Ensure 'name_embedding' exists and 'businessLine' column exists for description
            cur.execute(
                """
                SELECT id, "createdAt", name, status, "businessLine"
                FROM tickets
                WHERE "companyId" = %s AND name_embedding IS NOT NULL -- Assuming name_embedding for RAG search
                ORDER BY name_embedding <=> %s::vector
                LIMIT %s;
                """,
                (company_id_for_notes_tickets, query_embedding, limit)
            )
            tickets_snippets = cur.fetchall()
            print(f"Retrieved {len(tickets_snippets)} tickets for company {company_id}")
            for snippet in tickets_snippets:
                 retrieved_snippets.append({
                     'source': 'ticket',
                     'id': snippet['id'],
                     'createdAt': snippet['createdAt'],
                     'name': snippet['name'],
                     'status': snippet['status'],
                     'description': snippet.get('businessLine', 'N/A') # Use .get for safety
                 })

        return retrieved_snippets

    except Exception as e:
        st.error(f"Error retrieving relevant snippets via RAG: {e}")
        # Provide database connection status if known
        if conn is None:
             st.info("Database connection is not available for RAG retrieval.")
        return []

# ... (format_retrieved_snippets_for_llm remains the same) ...
def format_retrieved_snippets_for_llm(snippets):
    """Formats retrieved snippets from RAG into a string for the LLM prompt."""
    if not snippets:
        return "--- No relevant past conversations, notes, or tickets found ---\n\n"

    formatted_string = "--- Relevant Past Information (Retrieved via RAG) ---\n\n"

    # Sort snippets by date for better chronological flow if desired
    # Assuming 'createdAt' exists in all snippet types or handle missing dates
    # Use a robust key function that handles potential None or non-datetime types
    sorted_snippets = sorted(snippets, key=lambda x: x.get('createdAt', datetime.datetime.min) if isinstance(x.get('createdAt'), datetime.datetime) else datetime.datetime.min)


    for snippet in sorted_snippets:
        source = snippet.get('source', 'Unknown Source')
        item_id = snippet.get('id', 'N/A')
        created_at = snippet.get('createdAt', 'N/A')
        content = snippet.get('content', 'N/A') # Using 'content' as a fallback if specific field is missing

        # Adjust display based on source
        formatted_string += f"Source: {source} (ID: {item_id}, Created: {created_at})\n"

        if source == 'past_message':
             sender = snippet.get('sender', 'N/A')
             content_text = snippet.get('content', 'N/A') # Assuming 'content' holds the message text
             formatted_string += f"  Sender: {sender}\n"
             formatted_string += "  Content: " + textwrap.fill(str(content_text), width=80, subsequent_indent="  ") + "\n"
        elif source == 'note':
             author = snippet.get('author', 'N/A')
             content_text = snippet.get('content', 'N/A') # Assuming 'content' holds the note text
             formatted_string += f"  Author: {author}\n"
             formatted_string += "  Content: " + textwrap.fill(str(content_text), width=80, subsequent_indent="  ") + "\n"
        elif source == 'ticket':
             name = snippet.get('name', 'N/A')
             status = snippet.get('status', 'N/A')
             description = snippet.get('description', 'N/A')
             # For tickets, combine relevant fields into the 'content' display
             ticket_content_str = f"Name: {name}, Status: {status}"
             if description != 'N/A':
                  ticket_content_str += f", Description: {description}"
             formatted_string += "  Details: " + textwrap.fill(ticket_content_str, width=80, subsequent_indent="  ") + "\n"
             # Note: 'content' field in the ticket snippet dict might be unused,
             # using the specific ticket fields for display.
        else:
             # Fallback for unknown sources
             formatted_string += "  Content: " + textwrap.fill(str(content), width=80, subsequent_indent="  ") + "\n"

        formatted_string += "\n" # Add space between snippets

    formatted_string += "--- End Relevant Past Information ---\n\n"
    return formatted_string


# --- Streamlit App UI ---

st.title("📊 Agent Context Summarizer (Database RAG)")
st.markdown("Enter a Company ID to load context and enable RAG search on past data. Choose only either 285,306 or 447,688 or 558,916 for the sake of this MVP")

# --- Session State Initialization ---
# Ensure these are initialized early
if 'company_id_loaded' not in st.session_state:
    st.session_state['company_id_loaded'] = None
if 'general_company_context_string' not in st.session_state:
    st.session_state['general_company_context_string'] = ""
if 'messages_display' not in st.session_state:
    st.session_state['messages_display'] = []
if 'raw_general_context_data' not in st.session_state:
    st.session_state['raw_general_context_data'] = None


# --- Company ID Input ---
if st.session_state['company_id_loaded'] is None:
    st.subheader("Load Company Context")
    input_company_id = st.text_input("Enter Company ID:", key="company_id_input")

    # Only show the load button if the database connection is available
    if conn is not None:
        if st.button("Load Context"):
            if input_company_id:
                with st.spinner(f"Loading general context for Company ID: {input_company_id}..."):
                    raw_data = get_general_company_context(input_company_id)
                    if raw_data:
                        st.session_state['raw_general_context_data'] = raw_data
                        st.session_state['general_company_context_string'] = format_general_company_context_for_llm(raw_data)
                        st.session_state['company_id_loaded'] = input_company_id
                        st.session_state['messages_display'] = []
                        st.success(f"General context loaded for Company ID: {input_company_id}")
                        st.rerun() # Rerun to transition to chat view
                    else:
                        st.error(f"Could not load general context for Company ID: {input_company_id}. Please check the ID or database connection.")
            else:
                st.warning("Please enter a Company ID.")
    else:
        st.warning("Database connection not available. Cannot load company context.")


# --- Chat Interface (shown only after company_id is loaded and models are ready) ---
if st.session_state['company_id_loaded'] and chat_model is not None and embedding_model is not None and conn is not None:
    current_company_id = st.session_state['company_id_loaded']
    st.subheader(f"Chat for Company ID: {current_company_id} (Database RAG Enabled)")

    # Optional: Display loaded general context
    with st.expander("View Loaded General Company Context"):
        if st.session_state['general_company_context_string']:
             st.text(st.session_state['general_company_context_string'])
        else:
             st.info("No general context data available in session state.")

    # Display chat messages from history (these are just for display)
    for message in st.session_state['messages_display']:
         with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # Chat input
    prompt = st.chat_input("Ask about this company...")

    if prompt:
        # Add user message to display history
        st.session_state['messages_display'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- RAG Step: Generate query embedding and Retrieve relevant snippets ---
        with st.spinner("Generating query embedding and retrieving relevant data..."):
            query_embedding = generate_embedding_sync(prompt)

            if query_embedding is None:
                 st.error("Failed to generate embedding for your query. Cannot perform RAG.")
                 # Decide how to handle this - stop, or proceed without RAG?
                 # For now, let's just stop this turn and prevent LLM call.
                 st.session_state.messages_display[-1]['content'] += "\n\n*Error: Failed to generate query embedding.*"
                 st.experimental_rerun() # Rerun to show the error message


            # Retrieve relevant snippets from multiple sources using the query embedding
            relevant_snippets = retrieve_relevant_snippets_rag(current_company_id, query_embedding, limit=RETRIEVAL_LIMIT)
            formatted_relevant_snippets = format_retrieved_snippets_for_llm(relevant_snippets)

            # Optional: Show retrieved snippets to the agent for transparency/debugging
            # with st.sidebar:
            #      st.subheader("Retrieved Snippets (for LLM):")
            #      st.text(formatted_relevant_snippets)


        # --- Prepare the full prompt for the LLM ---
        llm_prompt_text = f"""
You are an AI assistant for an accounting firm, providing context and answering questions about client companies.
Use the provided General Company Context, the Relevant Past Information found via RAG, and the recent Chat History to answer the agent's question.
Prioritize information from the Relevant Past Information if it directly addresses the question.
If the answer is not found in the provided context, state that you don't have enough information.
Do not use external knowledge.

{st.session_state['general_company_context_string']}

{formatted_relevant_snippets}

--- Recent Chat History ---
{''.join([f'{m["role"].capitalize()}: {m["content"]}\n' for m in st.session_state['messages_display'][-6:]])} # Use last 6 turns from display history


---

Agent's Question: {prompt}
"""

        with st.spinner("Generating response..."):
            # Using st.chat_message("assistant") and a placeholder for streaming
            with st.chat_message("assistant"):
                 message_placeholder = st.empty() # Create an empty placeholder for the response
                 full_response_text = ""
                 try:
                     response = chat_model.generate_content(llm_prompt_text, stream=True)

                     # Stream the response into the placeholder
                     for chunk in response:
                         try:
                             if hasattr(chunk, 'text'):
                                 full_response_text += chunk.text
                                 message_placeholder.markdown(full_response_text + "▌") # Add cursor effect
                         except Exception as chunk_e:
                             print(f"Error processing chunk: {chunk_e}")
                             # Decide how to handle chunk errors - log, show partial response?
                             pass # Skip malformed chunks

                     message_placeholder.markdown(full_response_text) # Display final text without cursor

                 except Exception as e:
                     error_message = f"An error occurred during LLM generation: {e}"
                     st.error(error_message)
                     full_response_text = error_message # Store error as the response
                     message_placeholder.markdown(error_message) # Ensure error is displayed


        # Add assistant response (full text) to chat history
        st.session_state.messages_display.append({"role": "assistant", "content": full_response_text})


# --- Display status messages if components are not ready ---
if conn is None:
    st.warning("Waiting for database connection (check Cloud SQL Proxy status).")
if chat_model is None:
     st.warning("Waiting for chat model to initialize.")
if embedding_model is None:
     st.warning("Waiting for embedding model to initialize.")

# --- Optional: Proxy Status Display ---
# Add a small section to show proxy status for debugging
st.sidebar.subheader("Cloud SQL Proxy Status")
if st.session_state.get('cloudsql_proxy_process') is not None:
    process = st.session_state['cloudsql_proxy_process']
    if process.poll() is None:
        st.sidebar.success(f"Proxy Running (PID: {process.pid})")
        # Optional: Display last few lines of stderr for running process
        # This is tricky with Popen. You'd need to read non-blocking or periodically.
        # stderr_output = process.stderr.peek().decode() # peek might not be universally available or work as expected
        # st.sidebar.text("Recent STDERR:")
        # st.sidebar.text(stderr_output[-500:]) # Show last 500 chars
    else:
        st.sidebar.error(f"Proxy Exited (Return code: {process.returncode})")
        st.sidebar.text("Check logs above for details.")
        # Display captured output if available (this was already done on error)
else:
    st.sidebar.warning("Proxy process not started.")

st.sidebar.text(f"Temp key path: {st.session_state.get('cloudsql_temp_key_path', 'Not created')}")