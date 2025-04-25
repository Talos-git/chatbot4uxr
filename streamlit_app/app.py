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
import json
import atexit

# Import Vertex AI libraries for embedding
import vertexai
from vertexai.language_models import TextEmbeddingModel

# Import pgvector for psycopg2 type handling
from pgvector.psycopg2 import register_vector

# --- Configuration ---
# Access secrets from .streamlit/secrets.toml
# Ensure these secrets are configured correctly

# General GCP Project/Location for Vertex AI
GOOGLE_CLOUD_PROJECT = st.secrets["GOOGLE_CLOUD_PROJECT"]
GOOGLE_CLOUD_LOCATION = st.secrets["GOOGLE_CLOUD_LOCATION"]

# PostgreSQL Connection Details (application connects to the proxy)
# The proxy will listen on PG_HOST_APP:PG_PORT
# The application connects to the proxy at PG_HOST_APP:PG_PORT
# secrets.toml has host="127.0.0.1", which is correct for connecting to the proxy locally
PG_HOST_APP = st.secrets["postgres"]["host"] # Application connects to localhost proxy
PG_PORT = st.secrets["postgres"]["port"] # Local port the proxy listens on (e.g., 5432)

# Database connection details used by the *application* connecting through the proxy
PG_DATABASE = st.secrets["postgres"]["database"]
PG_USER = st.secrets["postgres"]["user"]
# Note: When using the proxy with --auto-iam-authn, the DB user should typically
# be an IAM user (like a service account email) configured in Cloud SQL.
# The password secret *might* not be needed in psycopg2.connect if IAM auth is fully used
# via the proxy, but we'll include it in the params for completeness if needed
# by your specific DB user setup.
PG_PASSWORD = st.secrets["postgres"]["password"]

# Cloud SQL Instance Connection Name (used by the proxy)
PG_INSTANCE_CONNECTION_NAME = st.secrets["postgres"]["instance_connection_name"]

# Gemini API Key
GENAI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Embedding Model Configuration (using Vertex AI)
EMBEDDING_MODEL_NAME = "text-embedding-005"
EMBEDDING_DIMENSION = 768
RETRIEVAL_LIMIT = 50

# Load the SA JSON dictionary directly from secrets
gcp_service_account_info = st.secrets["gcp"]


# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Agent Context Summarizer (DB RAG)", layout="wide")

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
atexit.register(cleanup_cloudsql_proxy)


def start_cloudsql_proxy(sa_info):
    """Starts the Cloud SQL Auth Proxy as a subprocess."""
    # Check if proxy is already running based on session state
    if st.session_state.get('cloudsql_proxy_process') is not None:
        if st.session_state['cloudsql_proxy_process'].poll() is None:
             # Check if the process is the one we expect (optional, but good)
             # You could try sending a signal or connecting to the admin port
             print("Cloud SQL Auth Proxy is already running and appears healthy.")
             return # Proxy is running, do nothing
        else:
             print("Cloud SQL Auth Proxy process found in state but is not running. Cleaning up state.")
             cleanup_cloudsql_proxy() # Clean up old state before restarting

    print("Attempting to start Cloud SQL Auth Proxy...")

    # --- 1. Save Service Account key info to a temporary file ---
    try:
        # Use mkstemp for better security than NamedTemporaryFile(delete=False)
        fd, temp_key_file_path = tempfile.mkstemp(suffix=".json", text=True)
        with os.fdopen(fd, 'w') as tmp:
             json.dump(sa_info, tmp)

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
    # Use --credentials-file to point to the temporary key file containing the SA key
    command = [
        proxy_executable,
        f"{PG_INSTANCE_CONNECTION_NAME}=tcp:{PG_HOST_APP}:{PG_PORT}", # e.g., "my-project:us-central1:my-instance=tcp:127.0.0.1:5432"
        "--auto-iam-authn",
        f"--credentials-file={st.session_state['cloudsql_temp_key_path']}",
        # "-verbose", # Optional: Uncomment for more proxy logging
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
            text=True, # Decode stdout/stderr as text
            # Consider setting a working directory if needed: cwd="."
        )
        st.session_state['cloudsql_proxy_process'] = process
        print(f"Cloud SQL Auth Proxy subprocess started with PID: {process.pid}")

        # Give the proxy a moment to start up and check for immediate failure
        startup_check_time = 5 # seconds
        start_time = time.time()
        proxy_failed_immediately = False
        while (time.time() - start_time) < startup_check_time:
             if process.poll() is not None: # Check if process has exited
                  proxy_failed_immediately = True
                  break # Exit the loop

             time.sleep(0.1) # Small sleep before checking again

        if proxy_failed_immediately:
             # Proxy failed to start, read stderr for errors
             stdout, stderr = process.communicate() # Read remaining output
             st.error(f"Cloud SQL Auth Proxy failed to start or terminated unexpectedly (Return code: {process.returncode}).")
             st.text(f"Proxy STDOUT:\n{stdout}")
             st.text(f"Proxy STDERR:\n{stderr}")
             cleanup_cloudsql_proxy() # Clean up the temp file and reset state
             return # Exit the startup function
        else:
             # If we reached here, the process is likely still running after the check time
             print("Cloud SQL Auth Proxy seems to be running after initial check.")
             # You might want to read the first few lines of stdout/stderr here too
             # print("Proxy Initial STDERR (if any):")
             # try:
             #      # Read non-blocking if possible or just a few lines
             #      # This is tricky with Popen.communicate() is better after poll()
             #      pass
             # except Exception as read_e:
             #      print(f"Error reading initial stderr: {read_e}")


    except Exception as e:
        st.error(f"An error occurred trying to start the Cloud SQL Auth Proxy subprocess: {e}")
        cleanup_cloudsql_proxy() # Clean up on failure
        return

    # Add a small delay after successful apparent start before attempting DB connection
    # to give the proxy time to open the port
    time.sleep(2) # Adjust if needed


# --- Call the function to start the proxy early in your script ---
# Pass the SA info dictionary to the startup function
# This ensures it runs when the app container starts or reruns without a cached proxy
# Use a button or initial load check to trigger it
if st.session_state['cloudsql_proxy_process'] is None:
     # You could add a button here, or just auto-start on first load attempt
     start_cloudsql_proxy(gcp_service_account_info)
     # Optional: st.rerun() could be used after starting the proxy


# --- Initialize Vertex AI and Embedding Model (cached) ---
@st.cache_resource
def initialize_vertex_ai_and_embedding_model(project, location, sa_info):
    """Initializes Vertex AI and loads the embedding model."""
    try:
        # Load credentials from the dictionary
        creds = service_account.Credentials.from_service_account_info(sa_info)
        vertexai.init(project=project, location=location, credentials=creds)
        print(f"Vertex AI initialized for project '{project}' in location '{location}'.")
        embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
        print(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded.")
        return embedding_model
    except Exception as e:
        st.error(f"Error initializing Vertex AI or loading embedding model: {e}")
        st.info("Please ensure:")
        st.info("- `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` are correct in secrets.")
        st.info(f"- The service account {sa_info.get('client_email', 'N/A')} has the `Vertex AI User` role.")
        st.info("- The text-embedding-005 model is available in your region.")
        # st.stop() # Consider stopping the app if this fails
        return None # Return None if initialization fails


# Pass the necessary info to the cached initialization function
embedding_model = initialize_vertex_ai_and_embedding_model(
    GOOGLE_CLOUD_PROJECT,
    GOOGLE_CLOUD_LOCATION,
    gcp_service_account_info
)


# --- Initialize LLM (cached) ---
@st.cache_resource
def get_generative_model(api_key):
     """Initializes and returns the generative model for chat."""
     try:
        if not api_key:
             st.warning("GEMINI_API_KEY is not set in Streamlit secrets. Chat model will not be available.")
             return None
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(LLM_MODEL_NAME)
     except Exception as e:
         st.error(f"Error configuring Gemini API for chat: {e}")
         # Provide troubleshooting steps for Gemini
         st.info("Please ensure:")
         st.info("- `GEMINI_API_KEY` is correctly set in `secrets.toml`.")
         st.info("- The key is valid and has access to the specified model (`gemini-2.0-flash`).")
         # st.stop() # Consider stopping if the chat model is critical
         return None # Return None if initialization fails

# Pass the API key to the cached function
chat_model = get_generative_model(GENAI_API_KEY)

# Add a check if the chat model failed to initialize
if chat_model is None:
    st.warning("Chat model is not available. Please check API key and configuration.")


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
         # Check if the proxy startup function was actually called
         if st.session_state.get('cloudsql_proxy_process') is None:
              st.error("Cloud SQL Auth Proxy was not started. Cannot connect to database.")
         else:
              st.error(f"Cloud SQL Auth Proxy is not running or has terminated (Return code: {st.session_state['cloudsql_proxy_process'].poll()}). Cannot connect to database.")
              st.info("Check the logs above for proxy startup errors.")
         # You could attempt to restart it here, but be careful with infinite loops
         # start_cloudsql_proxy(gcp_service_account_info)
         return None # Indicate failure

    try:
        print(f"Attempting database connection to {conn_params['host']}:{conn_params['port']} as user {conn_params['user']}...")
        conn = psycopg2.connect(**conn_params)
        # Register the vector type with psycopg2 using pgvector library
        register_vector(conn)
        print("pgvector.psycopg2.register_vector called.")
        print("Database connection successful!")
        return conn
    except Exception as e:
        st.error(f"Error connecting to database (check proxy status and connection details): {e}")
        # Provide more specific help for database connection issues
        st.info("Please ensure:")
        st.info(f"- The proxy is running and listening on {PG_HOST_APP}:{PG_PORT}.")
        st.info(f"- Database: '{PG_DATABASE}', User: '{PG_USER}' are correct.")
        st.info(f"- The database user '{PG_USER}' exists and is configured for IAM authentication in Cloud SQL.")
        st.info("- Your service account has the `Cloud SQL Client` role on the instance.")
        st.info("- The instance connection name is correct: `{}`".format(PG_INSTANCE_CONNECTION_NAME))
        # Attempt to read proxy logs if connection fails (might not get much here immediately)
        # print("\n--- Attempting to read recent proxy stderr ---")
        # try:
        #     process = st.session_state['cloudsql_proxy_process']
        #     if process and process.stderr:
        #          # This is tricky with Popen, may need different methods depending on how much output is buffered
        #          # process.stderr.peek() or process.stderr.read() might consume output.
        #          # For debugging, a simple read might work if the process exited.
        #          if process.poll() is not None: # Only try reading if process has exited
        #               stdout, stderr = process.communicate()
        #               print("Proxy STDOUT on exit:\n", stdout)
        #               print("Proxy STDERR on exit:\n", stderr)
        #          else:
        #               print("Proxy is running, live stderr reading is complex with Popen.")
        # except Exception as log_e:
        #     print(f"Error reading proxy stderr: {log_e}")
        return None

# Prepare connection parameters dictionary to pass to the cached function
# This includes only the parameters necessary for psycopg2.connect
db_connection_params = {
    "host": PG_HOST_APP, # Connect to the proxy's local address (127.0.0.1)
    "database": PG_DATABASE,
    "user": PG_USER,
    "port": PG_PORT, # Connect to the proxy's local port (5432)
    # Include password if your DB user requires it even with IAM proxy auth.
    # Standard IAM auth via proxy typically omits the password here.
    # "password": PG_PASSWORD
}

conn = get_db_connection(db_connection_params)

# Add a check if the database connection failed
if conn is None:
    st.error("Database connection failed. App functionality requiring DB will not work.")
    # Decide if you want to stop the app here
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
         # print("Skipping embedding for empty or whitespace text.") # Optional: log
         return np.zeros(EMBEDDING_DIMENSION).tolist() # Return zero vector for empty/whitespace

    try:
        # Use the loaded embedding model
        embeddings = embedding_model.get_embeddings([str(text)]) # get_embeddings expects a list
        return embeddings[0].values # Return the list of floats
    except Exception as e:
        st.error(f"Error generating embedding for text: '{str(text)[:50]}...' - {e}")
        # Provide more context
        st.info("Please ensure:")
        st.info("- `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` are correct.")
        st.info("- The service account has `Vertex AI User` role.")
        st.info("- The text-embedding-005 model is available in your region.")
        # Depending on severity, you might want to raise an exception or stop
        return None # Or handle gracefully


# --- Context Retrieval and Formatting Functions ---

def get_general_company_context(company_id):
    """Retrieves general context for a given company_id from structured tables."""
    # Get connection using the cached function, which also checks proxy status
    conn = get_db_connection(db_connection_params)
    if conn is None:
        return None # Cannot get context without connection

    context_data = {}
    # Ensure company_id is treated as a string for database queries if needed
    company_id_str = str(company_id)

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # --- 1. Company Information (using 'id' as PK) ---
            cur.execute('SELECT * FROM companies WHERE id = %s', (company_id_str,))
            company_info = cur.fetchone()
            if company_info:
                 context_data['company_info'] = dict(company_info)
            else:
                 st.warning(f"No company found for ID: {company_id_str}")
                 return None # Exit if the company doesn't exist

            # --- 2. Financial Years (using 'id' as PK and 'companyId' FK) ---
            # Assuming companyId in fiscalYears matches the format used in `companies.id`
            cur.execute('SELECT * FROM "fiscalYears" WHERE "companyId" = %s ORDER BY "endDate" DESC', (company_id_str,))
            context_data['fiscal_years'] = cur.fetchall()

            # Add other structured data retrieval here if needed (e.g., key contacts)

        return context_data

    except Exception as e:
        st.error(f"Error retrieving general company context from database: {e}")
        # Database connection might have failed after getting the connection object initially
        # Check conn status again if detailed error handling needed here
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
        # Ensure dictionary values are accessed robustly
        formatted_string += f"Company ID: {info.get('id', 'N/A')}\n"
        formatted_string += f"Company Name: {info.get('name', 'N/A')}\n"
        formatted_string += f"Status: {info.get('status', 'N/A')}\n"
        formatted_string += f"Type: {info.get('type', 'N/A')}\n"
        formatted_string += f"Functional Currency: {info.get('functionalCurrency', 'N/A')}\n"
        formatted_string += f"Tags: {info.get('tags', 'N/A')}\n"
        # Ensure dates are handled correctly if they aren't strings
        formatted_string += f"Opening Balance Date: {info.get('openingBalanceDate', 'N/A')}\n"
        formatted_string += f"Soft Lock Date: {info.get('softLockDate', 'N/A')}\n"
        formatted_string += f"Hard Lock Date: {info.get('hardLockDate', 'N/A')}\n"
        # Add other relevant company fields from your schema
        formatted_string += "\n"

    # Financial Years
    if 'fiscal_years' in context_data and context_data['fiscal_years']:
        formatted_string += "Fiscal Years:\n"
        for fy in context_data['fiscal_years']:
             # Adjust formatting based on your schema (e.g., id, endDate, rawData)
             # Ensure values are strings for formatting
            formatted_string += f"- ID: {fy.get('id', 'N/A')}, End Date: {fy.get('endDate', 'N/A')}, Raw Data Snippet: {str(fy.get('rawData', 'N/A'))[:100]}...\n" # Truncate raw data
        formatted_string += "\n"

    formatted_string += "--- End General Company Context ---\n\n"

    return formatted_string


def retrieve_relevant_snippets_rag(company_id, query_embedding, limit=RETRIEVAL_LIMIT):
    # Get connection using the cached function, which also checks proxy status
    conn = get_db_connection(db_connection_params)
    if conn is None:
        return [] # Cannot retrieve snippets without connection

    # Add check if query embedding failed
    if query_embedding is None:
        st.error("Query embedding is missing. Cannot perform RAG search.")
        return []

    retrieved_snippets = []

    # Ensure company_id is treated as a string for database queries if needed
    company_id_str = str(company_id)

    # Use the same company_id string for all tables for consistency,
    # unless you are certain they store the ID differently (e.g., one with, one without comma)
    # Based on your previous error showing "doggemgo:asia-southeast1:chatbot4uxr"
    # being passed to -instances, and the error showing the full string including comma,
    # it seems more likely that the companyId *in the DB tables* is the full string
    # with the comma. Let's use the original string value of input_company_id.
    # If the DB table *really* uses the string without the comma, you would need
    # to strip it here. Assuming the DB tables use the input company_id as-is.

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # --- Retrieve from past_messages ---
            # Assuming past_messages stores companyId as the full string with comma if input had one
            print(f"Querying past_messages for companyId: '{company_id_str}'")
            # Assuming 'message_from_who' and 'text' columns exist
            cur.execute(
                """
                SELECT id, "createdAt", message_from_who, text
                FROM past_messages
                WHERE "companyId" = %s AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (company_id_str, query_embedding, limit)
            )
            messages_snippets = cur.fetchall()
            print(f"Retrieved {len(messages_snippets)} past messages for company {company_id_str}")
            for snippet in messages_snippets:
                 retrieved_snippets.append({
                     'source': 'past_message',
                     'id': snippet.get('id'),
                     'createdAt': snippet.get('createdAt'),
                     'sender': snippet.get('message_from_who'),
                     'content': snippet.get('text')
                 })

            # --- Retrieve from notes ---
            # Assuming notes stores companyId as the full string with comma if input had one
            print(f"Querying notes for companyId: '{company_id_str}'")
            # Assuming 'lastModifiedByUserId' and 'text' columns exist
            cur.execute(
                """
                SELECT id, "createdAt", "lastModifiedByUserId", text
                FROM notes
                WHERE "companyId" = %s AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (company_id_str, query_embedding, limit)
            )
            notes_snippets = cur.fetchall()
            print(f"Retrieved {len(notes_snippets)} notes for company {company_id_str}")
            for snippet in notes_snippets:
                 retrieved_snippets.append({
                     'source': 'note',
                     'id': snippet.get('id'),
                     'createdAt': snippet.get('createdAt'),
                     'author': snippet.get('lastModifiedByUserId'),
                     'content': snippet.get('text')
                 })

            # --- Retrieve from tickets ---
            # Assuming tickets stores companyId as the full string with comma if input had one
            print(f"Querying tickets for companyId: '{company_id_str}'")
            # Assuming 'name_embedding', 'name', 'status', 'businessLine' columns exist
            cur.execute(
                """
                SELECT id, "createdAt", name, status, "businessLine"
                FROM tickets
                WHERE "companyId" = %s AND name_embedding IS NOT NULL -- Assuming name_embedding for RAG search
                ORDER BY name_embedding <=> %s::vector
                LIMIT %s;
                """,
                (company_id_str, query_embedding, limit)
            )
            tickets_snippets = cur.fetchall()
            print(f"Retrieved {len(tickets_snippets)} tickets for company {company_id_str}")
            for snippet in tickets_snippets:
                 retrieved_snippets.append({
                     'source': 'ticket',
                     'id': snippet.get('id'),
                     'createdAt': snippet.get('createdAt'),
                     'name': snippet.get('name'),
                     'status': snippet.get('status'),
                     'description': snippet.get('businessLine')
                 })

        # Filter out any snippets where essential data is missing (like content)
        retrieved_snippets = [s for s in retrieved_snippets if s.get('content') is not None or s.get('name') is not None]

        return retrieved_snippets

    except Exception as e:
        st.error(f"Error retrieving relevant snippets via RAG: {e}")
        # Provide database connection status if known
        if conn is None:
             st.info("Database connection is not available for RAG retrieval.")
        # Provide more specific help
        st.info("Please ensure:")
        st.info(f"- The company ID '{company_id_str}' exists in your database tables ('companies', 'past_messages', 'notes', 'tickets').")
        st.info("- The tables ('past_messages', 'notes', 'tickets') have non-NULL 'embedding' or 'name_embedding' columns for the records.")
        st.info("- Column names ('companyId', 'embedding', 'name_embedding', 'text', etc.) in your tables match the query.")
        return []


# ... (format_retrieved_snippets_for_llm remains the same) ...
def format_retrieved_snippets_for_llm(snippets):
    """Formats retrieved snippets from RAG into a string for the LLM prompt."""
    if not snippets:
        return "--- No relevant past conversations, notes, or tickets found ---\n\n"

    formatted_string = "--- Relevant Past Information (Retrieved via RAG) ---\n\n"

    # Sort snippets by date for better chronological flow if desired
    # Ensure 'createdAt' is handled if it's None or not a datetime
    sorted_snippets = sorted(snippets, key=lambda x: x.get('createdAt', datetime.datetime.min) if isinstance(x.get('createdAt'), datetime.datetime) else datetime.datetime.min)


    for snippet in sorted_snippets:
        source = snippet.get('source', 'Unknown Source')
        item_id = snippet.get('id', 'N/A')
        created_at = snippet.get('createdAt', 'N/A')

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
             if description is not None and description != 'N/A': # Check explicitly for None and 'N/A'
                  ticket_content_str += f", Description: {description}"
             formatted_string += "  Details: " + textwrap.fill(str(ticket_content_str), width=80, subsequent_indent="  ") + "\n"

        else: # Fallback for unknown sources, use the generic 'content' field if available
             content = snippet.get('content', 'N/A')
             formatted_string += "  Content: " + textwrap.fill(str(content), width=80, subsequent_indent="  ") + "\n"


        formatted_string += "\n" # Add space between snippets

    formatted_string += "--- End Relevant Past Information ---\n\n"
    return formatted_string


# --- Streamlit App UI ---

st.title("📊 Agent Context Summarizer (Database RAG)")
st.markdown("Enter a Company ID to load context and enable RAG search on past data. Choose only either 285,306 or 447,688 or 558,916 for the sake of this MVP")
st.warning("Ensure the Cloud SQL Auth Proxy executable is in your app's root directory and is executable (`chmod +x ./cloud-sql-proxy`).")


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
                        st.error(f"Could not load general context for Company ID: {input_company_id}. Please check the ID or database connection logs above.")
            else:
                st.warning("Please enter a Company ID.")
    else:
        st.warning("Database connection not available. Cannot load company context. Check Cloud SQL Proxy status and logs.")


# --- Chat Interface (shown only after company_id is loaded and models are ready) ---
# Check if ALL necessary components are ready
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
                 # Append error to the last message displayed and stop this turn
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
{''.join([f'{m["role"].capitalize()}: {m["content"]}\n' for m in st.session_state['messages_display'][-6:]])} # Use last N turns from display history


---

Agent's Question: {prompt}
"""

        with st.spinner("Generating response..."):
            # Using st.chat_message("assistant") and a placeholder for streaming
            with st.chat_message("assistant"):
                 message_placeholder = st.empty() # Create an empty placeholder for the response
                 full_response_text = ""
                 try:
                     # Ensure chat_model is not None before calling generate_content
                     if chat_model is None:
                          raise ValueError("Chat model is not initialized.")

                     response = chat_model.generate_content(llm_prompt_text, stream=True)

                     # Stream the response into the placeholder
                     for chunk in response:
                         try:
                             if hasattr(chunk, 'text'):
                                 full_response_text += chunk.text
                                 message_placeholder.markdown(full_response_text + "▌") # Add cursor effect
                         except Exception as chunk_e:
                             print(f"Error processing chunk: {chunk_e}")
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
# Display specific warnings if components failed to load
if conn is None:
    st.warning("Database connection failed. Check Cloud SQL Proxy status and logs.")
if chat_model is None:
     st.warning("Gemini Chat model not available. Check API key and configuration.")
if embedding_model is None:
     st.warning("Vertex AI Embedding model not available. Check GCP credentials and configuration.")


# --- Optional: Proxy Status Display ---
st.sidebar.subheader("Cloud SQL Proxy Status")
if st.session_state.get('cloudsql_proxy_process') is not None:
    process = st.session_state['cloudsql_proxy_process']
    if process.poll() is None:
        st.sidebar.success(f"Proxy Running (PID: {process.pid})")
        st.sidebar.text(f"Listening on 127.0.0.1:{PG_PORT}")
        st.sidebar.text(f"Instance: {PG_INSTANCE_CONNECTION_NAME}")
        # Optional: Add button to view recent logs (tricky with Popen)
    else:
        st.sidebar.error(f"Proxy Exited (Return code: {process.returncode})")
        st.sidebar.text("Check logs above for details.")
else:
    st.sidebar.warning("Proxy process not started.")

st.sidebar.text(f"Temp key path: {st.session_state.get('cloudsql_temp_key_path', 'Not created')}")
st.sidebar.text(f"DB Host (App): {PG_HOST_APP}")
st.sidebar.text(f"DB Port (App/Proxy): {PG_PORT}")
st.sidebar.text(f"DB User: {PG_USER}")
# st.sidebar.text(f"DB Password Provided: {'Yes' if PG_PASSWORD else 'No'}") # Careful with displaying presence of password
st.sidebar.text(f"DB Name: {PG_DATABASE}")
st.sidebar.text(f"Proxy Instance CN: {PG_INSTANCE_CONNECTION_NAME}")
st.sidebar.text(f"GCP Project: {GOOGLE_CLOUD_PROJECT}")
st.sidebar.text(f"GCP Location: {GOOGLE_CLOUD_LOCATION}")
st.sidebar.text(f"SA Email: {gcp_service_account_info.get('client_email', 'N/A')}")