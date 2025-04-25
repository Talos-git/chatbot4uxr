import streamlit as st
import psycopg2
from psycopg2 import extras
import google.generativeai as genai # Still used for the chat model
import numpy as np
import textwrap
import datetime
import os
import tempfile
import time # Used for potential delays if needed
from google.oauth2 import service_account

# --- Streamlit Page Configuration ---
# This must be the first Streamlit command
st.set_page_config(page_title="Agent Context Summarizer (DB RAG)", layout="wide")

# Import Vertex AI libraries for embedding
import vertexai
from vertexai.language_models import TextEmbeddingModel

# Import pgvector for psycopg2 type handling
from pgvector.psycopg2 import register_vector # Import the registration function

# Load the SA JSON from Streamlit secrets
creds = service_account.Credentials.from_service_account_info(st.secrets["gcp"])

# --- Configuration ---
# Access secrets from .streamlit/secrets.toml
# Ensure these secrets are configured correctly
GOOGLE_CLOUD_PROJECT = st.secrets["GOOGLE_CLOUD_PROJECT"]
GOOGLE_CLOUD_LOCATION = st.secrets["GOOGLE_CLOUD_LOCATION"]
PG_HOST = st.secrets["postgres"]["host"]
PG_DATABASE = st.secrets["postgres"]["database"]
PG_USER = st.secrets["postgres"]["user"]
PG_PASSWORD = st.secrets["postgres"]["password"]
PG_PORT = st.secrets["postgres"]["port"]

# LLM Configuration (for chat response)
# Using gemini-1.5-flash-latest for chat
LLM_MODEL_NAME = 'gemini-2.0-flash'

# Embedding Model Configuration (using Vertex AI)
EMBEDDING_MODEL_NAME = "text-embedding-005" # Use the model from your script
EMBEDDING_DIMENSION = 768 # Dimension for this model
RETRIEVAL_LIMIT = 50 # Number of relevant snippets to retrieve from EACH source (messages, notes, tickets)

# --- Initialize Vertex AI and Embedding Model (cached) ---
@st.cache_resource
def initialize_vertex_ai_and_embedding_model():
    """Initializes Vertex AI and loads the embedding model."""
    try:
        vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION, credentials=creds)
        print(f"Vertex AI initialized for project '{GOOGLE_CLOUD_PROJECT}' in location '{GOOGLE_CLOUD_LOCATION}'.")
        embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
        print(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded.")
        return embedding_model
    except Exception as e:
        st.error(f"Error initializing Vertex AI or loading embedding model: {e}")
        st.stop() # Stop if embedding model cannot be loaded

embedding_model = initialize_vertex_ai_and_embedding_model()


# --- Initialize LLM (cached) ---
@st.cache_resource
def get_generative_model():
     """Initializes and returns the generative model for chat."""
     try:
        # Using gemini-1.5-flash-latest for chat
        # Ensure your API key is set in Streamlit secrets for genai
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        return genai.GenerativeModel(LLM_MODEL_NAME)
     except Exception as e:
         st.error(f"Error configuring Gemini API for chat: {e}")
         st.stop()

chat_model = get_generative_model()


# --- Database Functions ---

@st.cache_resource # Cache the connection to reuse it across interactions
def get_db_connection():
    """Establishes and returns a database connection."""
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            database=PG_DATABASE,
            user=PG_USER,
            password=PG_PASSWORD,
            port=PG_PORT
        )
        # Register the vector type with psycopg2 using pgvector library
        register_vector(conn) # Uncommented this line
        print("pgvector.psycopg2.register_vector called.")
        return conn
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None

# Commented out the log_message function as requested
# def log_message(company_id, role, content):
#     """Logs a message to the past_messages table."""
#     conn = get_db_connection()
#     if conn is None:
#         return

#     try:
#         with conn.cursor() as cur:
#             # Generate embedding for the message content (this might be redundant if you have a DB trigger)
#             # If you rely solely on the script or a trigger, you can remove embedding generation here.
#             # However, embedding new messages as they come in is best practice for real-time RAG.
#             # Let's generate it here for real-time context update.
#             message_embedding = generate_embedding_sync(content) # Use sync embedding function

#             if message_embedding is None:
#                  st.warning(f"Could not generate embedding for message. Logging without embedding.")
#                  embedding_vector_str = None
#             else:
#                  embedding_vector_str = message_embedding # psycopg2 handles list -> PG vector

#             # Assuming 'past_messages' table with columns: id, companyId, role, text, createdAt, embedding
#             # Generate a unique ID (UUID recommended in real app)
#             message_id = str(datetime.datetime.now().timestamp()).replace('.', '') # Simple timestamp-based ID

#             cur.execute(
#                 'INSERT INTO past_messages (id, "companyId", role, text, "createdAt", embedding) VALUES (%s, %s, %s, %s, %s, %s)',
#                 (message_id, company_id, role, content, datetime.datetime.now(), embedding_vector_str)
#             )
#             conn.commit()
#     except Exception as e:
#         st.error(f"Error logging message to database: {e}")
#         if conn:
#             conn.rollback()

# --- Embedding Function (Synchronous for app.py) ---
# This is a synchronous version for use within the Streamlit app's request/response cycle
def generate_embedding_sync(text):
    """Generates an embedding for a single text using the Vertex AI model synchronously."""
    if not text or not str(text).strip():
         return np.zeros(EMBEDDING_DIMENSION).tolist()

    try:
        # Use the loaded embedding model
        embeddings = embedding_model.get_embeddings([str(text)]) # get_embeddings expects a list
        return embeddings[0].values # Return the list of floats
    except Exception as e:
        st.error(f"Error generating embedding for text: '{str(text)[:50]}...' - {e}")
        # Depending on severity, you might want to raise an exception
        return None # Or handle gracefully


# --- Context Retrieval and Formatting Functions ---

def get_general_company_context(company_id):
    """Retrieves general context for a given company_id from structured tables."""
    conn = get_db_connection()
    if conn is None:
        return None

    context_data = {}
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
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

            # Add other structured data retrieval here if needed (e.g., key contacts from a users table if you re-add one)

        return context_data

    except Exception as e:
        st.error(f"Error retrieving general company context from database: {e}")
        return None

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
        for fy in context_data['fiscal_years']:
             # Adjust formatting based on your schema (e.g., id, endDate, rawData)
            formatted_string += f"- ID: {fy.get('id', 'N/A')}, End Date: {fy.get('endDate', 'N/A')}, Raw Data Snippet: {str(fy.get('rawData', 'N/A'))[:100]}...\n" # Truncate raw data
        formatted_string += "\n"

    formatted_string += "--- End General Company Context ---\n\n"

    return formatted_string

def retrieve_relevant_snippets_rag(company_id, query_embedding, limit=RETRIEVAL_LIMIT):
    conn = get_db_connection()
    if conn is None:
        return []

    retrieved_snippets = []

    # Create a version of the company_id without the comma for past_messages query
    # Assumes the input company_id string might contain a comma
    company_id_for_past_messages = company_id.replace(',', '')

    # Use the original company_id (which might contain a comma) for notes and tickets query
    # Assumes notes and tickets tables store companyId with the comma
    company_id_for_notes_tickets = company_id # Use the original input value

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # --- Retrieve from past_messages ---
            print(f"Querying past_messages for companyId (no comma): '{company_id_for_past_messages}'")
            cur.execute(
                """
                SELECT id, "createdAt", message_from_who, text
                FROM past_messages
                WHERE "companyId" = %s AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector -- Explicit cast here
                LIMIT %s;
                """,
                (company_id_for_past_messages, query_embedding, limit) # Use the variable WITHOUT the comma here
            )
            messages_snippets = cur.fetchall()
            print(f"Retrieved {len(messages_snippets)} past messages for company {company_id}")
            for snippet in messages_snippets:
                 retrieved_snippets.append({
                     'source': 'past_message',
                     'id': snippet['id'],
                     'createdAt': snippet['createdAt'],
                     'sender': snippet['message_from_who'],
                     'content': snippet['text']
                 })

            # --- Retrieve from notes ---
            print(f"Querying notes for companyId (with comma): '{company_id_for_notes_tickets}'")
            cur.execute(
                """
                SELECT id, "createdAt", "lastModifiedByUserId", text
                FROM notes
                WHERE "companyId" = %s AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector -- Explicit cast here
                LIMIT %s;
                """,
                (company_id_for_notes_tickets, query_embedding, limit) # Use the variable WITH the comma here
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
            cur.execute(
                """
                SELECT id, "createdAt", name, status, "businessLine"
                FROM tickets
                WHERE "companyId" = %s AND name_embedding IS NOT NULL
                ORDER BY name_embedding <=> %s::vector -- Explicit cast here
                LIMIT %s;
                """,
                (company_id_for_notes_tickets, query_embedding, limit) # Use the variable WITH the comma here
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
                     'description': snippet.get('businessLine')
                 })

        return retrieved_snippets

    except Exception as e:
        st.error(f"Error retrieving relevant snippets via RAG: {e}")
        return []
    
def format_retrieved_snippets_for_llm(snippets):
    """Formats retrieved snippets from RAG into a string for the LLM prompt."""
    if not snippets:
        return "--- No relevant past conversations, notes, or tickets found ---\n\n"

    formatted_string = "--- Relevant Past Information (Retrieved via RAG) ---\n\n"

    # Sort snippets by date for better chronological flow if desired
    # Assuming 'createdAt' exists in all snippet types or handle missing dates
    sorted_snippets = sorted(snippets, key=lambda x: x.get('createdAt', datetime.datetime.min))

    for snippet in sorted_snippets:
        source = snippet.get('source', 'Unknown Source')
        item_id = snippet.get('id', 'N/A')
        created_at = snippet.get('createdAt', 'N/A')
        content = snippet.get('content', 'N/A')

        formatted_string += f"Source: {source} (ID: {item_id}, Created: {created_at})\n"

        if source == 'past_message':
             sender = snippet.get('sender', 'N/A')
             formatted_string += f"  Sender: {sender}\n"
             formatted_string += "  Content: " + textwrap.fill(content, width=80, subsequent_indent="  ") + "\n"
        elif source == 'note':
             author = snippet.get('author', 'N/A')
             formatted_string += f"  Author: {author}\n"
             formatted_string += "  Content: " + textwrap.fill(content, width=80, subsequent_indent="  ") + "\n"
        elif source == 'ticket':
             name = snippet.get('name', 'N/A')
             status = snippet.get('status', 'N/A')
             description = snippet.get('description', 'N/A')
             formatted_string += f"  Name: {name}\n"
             formatted_string += f"  Status: {status}\n"
             if description != 'N/A':
                 formatted_string += "  Description: " + textwrap.fill(description, width=80, subsequent_indent="  ") + "\n"
             # Note: For tickets, you might primarily embed/search the name/description,
             # but retrieve/display other relevant ticket fields like status, assignee, etc.
        else:
             # Fallback for unknown sources
             formatted_string += "  Content: " + textwrap.fill(content, width=80, subsequent_indent="  ") + "\n"

        formatted_string += "\n" # Add space between snippets

    formatted_string += "--- End Relevant Past Information ---\n\n"
    return formatted_string


# --- Streamlit App UI ---

st.title("📊 Agent Context Summarizer (Database RAG)")
st.markdown("Enter a Company ID to load context and enable RAG search on past data. Choose only either 285,306 or 447,688 or 558,916 for the sake of this MVP")

# --- Session State Initialization ---
# Moved initialization to appear before first use
if 'company_id_loaded' not in st.session_state:
    st.session_state['company_id_loaded'] = None
if 'general_company_context_string' not in st.session_state:
    st.session_state['general_company_context_string'] = "" # This holds the formatted *initial* context
if 'messages_display' not in st.session_state:
    # Messages for display in the chat UI (recent history for conversational flow)
    st.session_state['messages_display'] = []
if 'raw_general_context_data' not in st.session_state:
    st.session_state['raw_general_context_data'] = None


# --- Company ID Input ---
if st.session_state['company_id_loaded'] is None:
    st.subheader("Load Company Context")
    input_company_id = st.text_input("Enter Company ID:", key="company_id_input")

    if st.button("Load Context"):
        if input_company_id:
            with st.spinner(f"Loading general context for Company ID: {input_company_id}..."):
                raw_data = get_general_company_context(input_company_id)
                if raw_data:
                    st.session_state['raw_general_context_data'] = raw_data # Store raw data
                    st.session_state['general_company_context_string'] = format_general_company_context_for_llm(raw_data)
                    st.session_state['company_id_loaded'] = input_company_id
                    st.session_state['messages_display'] = [] # Clear display messages for the new company
                    st.success(f"General context loaded for Company ID: {input_company_id}")
                    # Note: RAG snippets are NOT loaded here, they are retrieved PER QUERY
                    # st.rerun() # Rerun to transition to chat view - optional but clean
                else:
                    st.error(f"Could not load general context for Company ID: {input_company_id}. Please check the ID or database connection.")
        else:
            st.warning("Please enter a Company ID.")

# --- Chat Interface (shown only after company_id is loaded) ---
if st.session_state['company_id_loaded']:
    current_company_id = st.session_state['company_id_loaded']
    st.subheader(f"Chat for Company ID: {current_company_id} (Database RAG Enabled)")

    # Optional: Display loaded general context
    with st.expander("View Loaded General Company Context"):
        if st.session_state['general_company_context_string']:
             st.text(st.session_state['general_company_context_string'])
        else:
             st.info("No general context data available in session state.")

    # Display chat messages from history (these are just for display)
    # The *actual* historical messages used by the LLM for RAG are retrieved dynamically
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

        # Commented out logging user message to DB as requested
        # log_message(current_company_id, "user", prompt)

        # --- RAG Step: Generate query embedding and Retrieve relevant snippets ---
        with st.spinner("Generating query embedding and retrieving relevant data..."):
            query_embedding = generate_embedding_sync(prompt)

            if query_embedding is None:
                 st.error("Failed to generate embedding for your query. Cannot perform RAG.")
                 # Decide how to handle this - stop, or proceed without RAG?
                 # For now, let's stop this turn.
                 st.stop()

            # Retrieve relevant snippets from multiple sources using the query embedding
            relevant_snippets = retrieve_relevant_snippets_rag(current_company_id, query_embedding, limit=RETRIEVAL_LIMIT)
            formatted_relevant_snippets = format_retrieved_snippets_for_llm(relevant_snippets)

            # Optional: Show retrieved snippets to the agent for transparency/debugging
            # with st.sidebar:
            #      st.subheader("Retrieved Snippets (for LLM):")
            #      st.text(formatted_relevant_snippets)


        # --- Prepare the full prompt for the LLM ---
        # This includes:
        # 1. System Instructions
        # 2. General Company Context (loaded upfront)
        # 3. Retrieved Relevant Snippets (via RAG for the current query)
        # 4. Recent Chat History (from the *current* display session state)
        # 5. Current Agent's Question

        # Get recent chat history from session state for conversational flow
        # Limit history to avoid exceeding context window
        # Adjust the number of turns based on LLM context window and total context size
        recent_chat_history_for_llm = st.session_state['messages_display'][-6:] # Example: last 6 turns (user+model pairs)

        llm_prompt_text = f"""
You are an AI assistant for an accounting firm, providing context and answering questions about client companies.
Use the provided General Company Context, the Relevant Past Information found via RAG, and the recent Chat History to answer the agent's question.
Prioritize information from the Relevant Past Information if it directly addresses the question.
If the answer is not found in the provided context, state that you don't have enough information.
Do not use external knowledge.

{st.session_state['general_company_context_string']}

{formatted_relevant_snippets}

--- Recent Chat History ---
{''.join([f'{m["role"].capitalize()}: {m["content"]}\n' for m in recent_chat_history_for_llm])}

---

Agent's Question: {prompt}
"""

        with st.spinner("Generating response..."):
            # Send the full prompt to the LLM
            # Using generate_content with the full prompt string
            try:
                response = chat_model.generate_content(llm_prompt_text, stream=True)

                # Define a generator function to yield only the text parts from chunks
                def text_chunk_generator(stream):
                    """Yields text from stream chunks, handling potential errors."""
                    for chunk in stream:
                        try:
                            if hasattr(chunk, 'text'):
                                yield chunk.text
                        except Exception as e:
                            # Handle cases where a chunk might not have text or is malformed
                            print(f"Error processing chunk: {e}") # Log error to console
                            pass

                # Use st.write_stream with the text-only generator
                full_response_text = st.write_stream(text_chunk_generator(response))

            except Exception as e:
                error_message = f"An error occurred during LLM generation: {e}"
                st.error(error_message)
                full_response_text = error_message # Store error as the response for logging/history
                # Ensure the error is displayed if st.write_stream failed
                message_placeholder.markdown(error_message)


        # 3. Add assistant response (full text) to chat history and log it
        # Commented out logging assistant message to DB as requested
        # timestamp_assistant = datetime.datetime.now().isoformat()
        # log_message(current_company_id, "model", full_response_text)
        # Add the full response text to the display history
        st.session_state.messages_display.append({"role": "assistant", "content": full_response_text})

    # (Optional) Rerun if needed, though Streamlit usually handles reruns on widget interaction
    # st.rerun()
