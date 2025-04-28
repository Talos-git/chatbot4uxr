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
from google.oauth2 import service_account
import shlex

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
PG_PASSWORD = st.secrets["postgres"]["password"] # Uncomment if password is required by your DB user

# Cloud SQL Instance Connection Name (used by the proxy)
PG_INSTANCE_CONNECTION_NAME = st.secrets["postgres"]["instance_connection_name"]

# Gemini API Key
GENAI_API_KEY = st.secrets["GEMINI_API_KEY"]

LLM_MODEL_NAME = 'gemini-2.0-flash'

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


def start_cloudsql_proxy(sa_info_attrdict):
    """Starts the Cloud SQL Auth Proxy as a subprocess."""
    # Check if proxy is already running based on session state
    if st.session_state.get('cloudsql_proxy_process') is not None:
        if st.session_state['cloudsql_proxy_process'].poll() is None:
             print("Cloud SQL Auth Proxy is already running and appears healthy.")
             return # Proxy is running, do nothing
        else:
             print("Cloud SQL Auth Proxy process found in state but is not running. Cleaning up state.")
             cleanup_cloudsql_proxy() # Clean up old state before restarting

    print("Attempting to start Cloud SQL Auth Proxy...")

    # --- 1. Save Service Account key info to a temporary file ---
    try:
        sa_info_dict = dict(sa_info_attrdict)
        # Use mkstemp for better security than NamedTemporaryFile(delete=False)
        fd, temp_key_file_path = tempfile.mkstemp(suffix=".json", text=True)
        with os.fdopen(fd, 'w') as tmp:
             json.dump(sa_info_dict, tmp)

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
        PG_INSTANCE_CONNECTION_NAME,
        "--address", PG_HOST_APP,
        "--port", str(PG_PORT),
        #f"{PG_INSTANCE_CONNECTION_NAME}=tcp:{PG_HOST_APP}:{PG_PORT}", # e.g., "my-project:us-central1:my-instance=tcp:127.0.0.1:5432"
        "--auto-iam-authn",
        f"--credentials-file={st.session_state['cloudsql_temp_key_path']}",
        # "-verbose", # Optional: Uncomment for more proxy logging
    ]

    # --- Add this block within start_cloudsql_proxy ---
    # Attempt to kill any lingering proxy processes for this specific instance and port
    print(f"Attempting to kill any existing proxy processes using port {PG_PORT}...")
    try:
        # Construct a pkill command targeting the specific proxy instance and port
        # Uses -f to match against the full command line argument string
        # Quotes are important for safety with shlex.quote
        instance_name_quoted = shlex.quote(PG_INSTANCE_CONNECTION_NAME)
        port_quoted = shlex.quote(str(PG_PORT))
        # This pattern tries to be specific to avoid killing unrelated processes
        pkill_pattern = f"cloud-sql-proxy.*{instance_name_quoted}.*--port {port_quoted}"
        pkill_command = ["pkill", "-f", pkill_pattern]

        print(f"Running cleanup command: {' '.join(pkill_command)}")
        # Run pkill. We don't check=True because it's okay if no process was found (it returns non-zero).
        kill_result = subprocess.run(pkill_command, capture_output=True, text=True, check=False)

        if kill_result.returncode == 0:
            print(f"Successfully sent kill signal to matching processes.")
        elif "no process found" in kill_result.stderr.lower() or kill_result.returncode == 1:
            print("No lingering proxy processes found matching the pattern.")
        else:
            # Log if pkill failed for other reasons, but proceed anyway
            print(f"Warning: pkill command exited with code {kill_result.returncode}. Stderr: {kill_result.stderr.strip()}")

        # Give the OS a moment to release the port after killing
        time.sleep(10) # Sleep for 1 second

    except FileNotFoundError:
        print("Warning: 'pkill' command not found. Cannot perform automatic cleanup of lingering processes.")
    except Exception as kill_e:
        print(f"An error occurred during the proxy cleanup attempt: {kill_e}")
    # --- End of added block ---

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
if st.session_state['cloudsql_proxy_process'] is None:
     start_cloudsql_proxy(gcp_service_account_info)


# --- Initialize Vertex AI and Embedding Model (cached) ---
# FIX: Prefix the unhashable argument `sa_info` with an underscore
@st.cache_resource
def initialize_vertex_ai_and_embedding_model(project, location, _sa_info):
    """Initializes Vertex AI and loads the embedding model."""
    try:
        # Load credentials from the dictionary passed as _sa_info
        creds = service_account.Credentials.from_service_account_info(_sa_info)
        vertexai.init(project=project, location=location, credentials=creds)
        print(f"Vertex AI initialized for project '{project}' in location '{location}'.")
        embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
        print(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded.")
        return embedding_model
    except Exception as e:
        st.error(f"Error initializing Vertex AI or loading embedding model: {e}")
        st.info("Please ensure:")
        st.info("- `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` are correct in secrets.")
        # Safely access client_email from _sa_info dictionary if available
        sa_email = _sa_info.get('client_email', 'N/A') if isinstance(_sa_info, dict) else 'N/A'
        st.info(f"- The service account {sa_email} has the `Vertex AI User` role.")
        st.info("- The text-embedding-005 model is available in your region.")
        # st.stop() # Consider stopping the app if this fails
        return None # Return None if initialization fails


# Pass the necessary info to the cached initialization function
# The call site remains the same, only the function definition changes
embedding_model = initialize_vertex_ai_and_embedding_model(
    GOOGLE_CLOUD_PROJECT,
    GOOGLE_CLOUD_LOCATION,
    gcp_service_account_info # This is the AttrDict, passed to _sa_info
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
# FIX: Prefix the unhashable argument `conn_params` with an underscore
@st.cache_resource(hash_funcs={
    type(st.session_state): lambda _: None, # Don't hash the whole session state
    subprocess.Popen: lambda _: None # Don't hash the process object
    }, show_spinner="Connecting to database...")
def get_db_connection(_conn_params):
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
        # Use the _conn_params dictionary to establish the connection
        print(f"Attempting database connection to {_conn_params['host']}:{_conn_params['port']} as user {_conn_params['user']}...")
        conn = psycopg2.connect(**_conn_params)
        # Register the vector type with psycopg2 using pgvector library
        register_vector(conn)
        print("pgvector.psycopg2.register_vector called.")
        print("Database connection successful!")
        return conn
    except Exception as e:
        st.error(f"Error connecting to database (check proxy status and connection details): {e}")
        st.info("Please ensure:")
        st.info(f"- The proxy is running and listening on {PG_HOST_APP}:{PG_PORT}.")
        st.info(f"- Database: '{PG_DATABASE}', User: '{PG_USER}' are correct.")
        # Referencing _conn_params directly for password check might be risky if key isn't always there
        # Using the global PG_PASSWORD for the hint text instead
        st.info(f"- The database user '{PG_USER}' exists and is configured for IAM authentication in Cloud SQL (if not using password).")
        st.info(f"- Your service account has the `Cloud SQL Client` role on the instance.")
        st.info(f"- The instance connection name is correct: `{PG_INSTANCE_CONNECTION_NAME}`")

        return None

# Prepare connection parameters dictionary to pass to the cached function
db_connection_params = {
    "host": PG_HOST_APP, # Connect to the proxy's local address (127.0.0.1)
    "database": PG_DATABASE,
    "user": PG_USER,
    "port": PG_PORT, # Connect to the proxy's local port (5432)
    # Include password if your DB user requires it even with IAM proxy auth.
    # Standard IAM auth via proxy typically omits the password here.
    "password": PG_PASSWORD # Uncommented based on provided secrets, but review if needed for IAM auth setup
}

# Pass the dictionary to the cached function
# The call site remains the same, only the function definition changes
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
         return np.zeros(EMBEDDING_DIMENSION).tolist() # Return zero vector for empty/whitespace

    try:
        # Use the loaded embedding model
        embeddings = embedding_model.get_embeddings([str(text)]) # get_embeddings expects a list
        return embeddings[0].values # Return the list of floats
    except Exception as e:
        st.error(f"Error generating embedding for text: '{str(text)[:50]}...' - {e}")
        st.info("Please ensure:")
        st.info("- `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` are correct.")
        st.info("- The service account has `Vertex AI User` role.")
        st.info("- The text-embedding-005 model is available in your region.")
        return None


# --- Context Retrieval and Formatting Functions ---

def get_general_company_context(company_id):
    """Retrieves general context for a given company_id from structured tables."""
    # Get connection using the cached function, which also checks proxy status
    conn = get_db_connection(db_connection_params) # Pass params to allow cache check
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
        formatted_string += f"Company ID: {info.get('id', 'N/A')}\n"
        formatted_string += f"Company Name: {info.get('name', 'N/A')}\n"
        formatted_string += f"Status: {info.get('status', 'N/A')}\n"
        formatted_string += f"Type: {info.get('type', 'N/A')}\n"
        formatted_string += f"Functional Currency: {info.get('functionalCurrency', 'N/A')}\n"
        formatted_string += f"Tags: {info.get('tags', 'N/A')}\n"
        formatted_string += f"Opening Balance Date: {info.get('openingBalanceDate', 'N/A')}\n"
        formatted_string += f"Soft Lock Date: {info.get('softLockDate', 'N/A')}\n"
        formatted_string += f"Hard Lock Date: {info.get('hardLockDate', 'N/A')}\n"
        formatted_string += "\n"

    # Financial Years
    if 'fiscal_years' in context_data and context_data['fiscal_years']:
        formatted_string += "Fiscal Years:\n"
        for fy in context_data['fiscal_years']:
             formatted_string += f"- ID: {fy.get('id', 'N/A')}, End Date: {fy.get('endDate', 'N/A')}, Raw Data Snippet: {str(fy.get('rawData', 'N/A'))[:100]}...\n" # Truncate raw data
        formatted_string += "\n"

    formatted_string += "--- End General Company Context ---\n\n"

    return formatted_string


def retrieve_relevant_snippets_rag(company_id, query_embedding, limit=RETRIEVAL_LIMIT):
    # Get connection using the cached function, which also checks proxy status
    conn = get_db_connection(db_connection_params) # Pass params to allow cache check
    if conn is None:
        return [] # Cannot retrieve snippets without connection

    # Add check if query embedding failed
    if query_embedding is None:
        st.error("Query embedding is missing. Cannot perform RAG search.")
        return []

    retrieved_snippets = []

    # Ensure company_id is treated as a string for database queries
    company_id_str = str(company_id)

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # --- Retrieve from past_messages ---
            print(f"Querying past_messages for companyId: '{company_id_str}'")
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
            print(f"Querying notes for companyId: '{company_id_str}'")
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
            print(f"Querying tickets for companyId: '{company_id_str}'")
            cur.execute(
                """
                SELECT id, "createdAt", name, status, "businessLine"
                FROM tickets
                WHERE "companyId" = %s AND name_embedding IS NOT NULL
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

        # Filter out any snippets where essential data is missing
        # Assuming 'content' is essential for past_message/note, 'name' for ticket
        retrieved_snippets = [s for s in retrieved_snippets if (s.get('source') in ['past_message', 'note'] and s.get('content')) or (s.get('source') == 'ticket' and s.get('name'))]


        return retrieved_snippets

    except Exception as e:
        st.error(f"Error retrieving relevant snippets via RAG: {e}")
        if conn is None:
             st.info("Database connection is not available for RAG retrieval.")
        st.info("Please ensure:")
        st.info(f"- The company ID '{company_id_str}' exists in your database tables.")
        st.info("- Tables ('past_messages', 'notes', 'tickets') have non-NULL embedding columns.")
        st.info("- Column names in your tables match the queries.")
        return []

# ... (format_retrieved_snippets_for_llm remains the same) ...
def format_retrieved_snippets_for_llm(snippets):
    """Formats retrieved snippets from RAG into a string for the LLM prompt."""
    if not snippets:
        return "--- No relevant past conversations, notes, or tickets found ---\n\n"

    formatted_string = "--- Relevant Past Information (Retrieved via RAG) ---\n\n"

    # Sort snippets by date for better chronological flow if desired
    # Handle potential None or non-datetime types for 'createdAt'
    sorted_snippets = sorted(snippets, key=lambda x: x.get('createdAt', datetime.datetime.min) if isinstance(x.get('createdAt'), datetime.datetime) else datetime.datetime.min)

    for snippet in sorted_snippets:
        source = snippet.get('source', 'Unknown Source')
        item_id = snippet.get('id', 'N/A')
        created_at = snippet.get('createdAt', 'N/A')

        formatted_string += f"Source: {source} (ID: {item_id}, Created: {created_at})\n"

        if source == 'past_message':
             sender = snippet.get('sender', 'N/A')
             content_text = snippet.get('content', 'N/A')
             formatted_string += f"  Sender: {sender}\n"
             formatted_string += "  Content: " + textwrap.fill(str(content_text), width=80, subsequent_indent="  ") + "\n"
        elif source == 'note':
             author = snippet.get('author', 'N/A')
             content_text = snippet.get('content', 'N/A')
             formatted_string += f"  Author: {author}\n"
             formatted_string += "  Content: " + textwrap.fill(str(content_text), width=80, subsequent_indent="  ") + "\n"
        elif source == 'ticket':
             name = snippet.get('name', 'N/A')
             status = snippet.get('status', 'N/A')
             description = snippet.get('description', 'N/A')
             ticket_content_str = f"Name: {name}, Status: {status}"
             if description is not None and description != 'N/A':
                  ticket_content_str += f", Description: {description}"
             formatted_string += "  Details: " + textwrap.fill(str(ticket_content_str), width=80, subsequent_indent="  ") + "\n"
        else: # Fallback for unknown sources
             content = snippet.get('content', 'N/A')
             formatted_string += "  Content: " + textwrap.fill(str(content), width=80, subsequent_indent="  ") + "\n"

        formatted_string += "\n"

    formatted_string += "--- End Relevant Past Information ---\n\n"
    return formatted_string


# --- Streamlit App UI ---

st.title("📊 Agent Context Summarizer (Database RAG)")
st.markdown("Enter a Company ID to load context and enable RAG search on past data. Choose only either 285,306 or 447,688 or 558,916 for the sake of this MVP")
st.warning("Ensure the Cloud SQL Auth Proxy executable is in your app's root directory and is executable (`chmod +x ./cloud-sql-proxy`).")


# --- Session State Initialization ---
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

    # Only show the load button if essential components are ready
    if conn is not None and chat_model is not None and embedding_model is not None:
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
        st.warning("App components are not fully initialized. Cannot load company context. Check logs for errors.")


# --- Chat Interface (shown only after company_id is loaded and models are ready) ---
# Check if ALL necessary components are ready before showing chat
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
                 st.session_state.messages_display[-1]['content'] += "\n\n*Error: Failed to generate query embedding.*"
                 st.experimental_rerun()


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
{''.join([f'{m["role"].capitalize()}: {m["content"]}\n' for m in st.session_state['messages_display'][-6:]])}


---

Agent's Question: {prompt}
"""

        with st.spinner("Generating response..."):
            with st.chat_message("assistant"):
                 message_placeholder = st.empty()
                 full_response_text = ""
                 try:
                     if chat_model is None:
                          raise ValueError("Chat model is not initialized.")

                     response = chat_model.generate_content(llm_prompt_text, stream=True)

                     for chunk in response:
                         try:
                             if hasattr(chunk, 'text'):
                                 full_response_text += chunk.text
                                 message_placeholder.markdown(full_response_text + "▌")
                         except Exception as chunk_e:
                             print(f"Error processing chunk: {chunk_e}")
                             pass

                     message_placeholder.markdown(full_response_text)

                 except Exception as e:
                     error_message = f"An error occurred during LLM generation: {e}"
                     st.error(error_message)
                     full_response_text = error_message
                     message_placeholder.markdown(error_message)


        st.session_state.messages_display.append({"role": "assistant", "content": full_response_text})


# --- Display status messages if components are not ready ---
# These messages are shown initially or if something fails during setup
if conn is None:
    st.warning("Database connection failed. Please check Cloud SQL Proxy status and logs.")
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
        st.sidebar.text(f"Listening on {PG_HOST_APP}:{PG_PORT}")
        st.sidebar.text(f"Instance: {PG_INSTANCE_CONNECTION_NAME}")
    else:
        st.sidebar.error(f"Proxy Exited (Return code: {process.returncode})")
        st.sidebar.text("Check logs above for details.")
else:
    st.sidebar.warning("Proxy process not started.")

st.sidebar.text(f"Temp key path: {st.session_state.get('cloudsql_temp_key_path', 'Not created')}")
st.sidebar.text(f"DB Host (App): {PG_HOST_APP}")
st.sidebar.text(f"DB Port (App/Proxy): {PG_PORT}")
st.sidebar.text(f"DB User: {PG_USER}")
# st.sidebar.text(f"DB Password Provided: {'Yes' if 'password' in db_connection_params else 'No'}") # Safer check
st.sidebar.text(f"DB Name: {PG_DATABASE}")
st.sidebar.text(f"Proxy Instance CN: {PG_INSTANCE_CONNECTION_NAME}")
st.sidebar.text(f"GCP Project: {GOOGLE_CLOUD_PROJECT}")
st.sidebar.text(f"GCP Location: {GOOGLE_CLOUD_LOCATION}")
st.sidebar.text(f"SA Email: {gcp_service_account_info.get('client_email', 'N/A')}")