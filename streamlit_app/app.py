import streamlit as st
import google.generativeai as genai
import pandas as pd
import datetime
import os
import time # Required for file processing check delay
import csv # Explicitly import csv for logging

# --- Configuration ---
LOG_FILE = "chat_log_streaming.csv"
# Make sure this CSV file exists and contains your knowledge data.
# Columns could be: client_name, user_role, date, message_text, ticket_id, ticket_status, etc.
KNOWLEDGE_CSV_PATH = "your_knowledge.csv"

# --- Helper Functions ---

def setup_gemini_client():
    """Sets up the Gemini client using API key from Streamlit secrets."""
    try:
        # Recommended: Set API Key in Streamlit secrets (secrets.toml)
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        # Use a model that supports File API and is good for chat
        # gemini-1.5-flash is faster and cheaper, good for MVP
        return genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}")
        st.stop() # Stop execution if API key is missing/invalid

def upload_knowledge_file(file_path):
    """Uploads the knowledge CSV to Gemini File API if not already done in session."""
    # Check if we've already uploaded and got an active file object in this session
    if "knowledge_file_active" in st.session_state and st.session_state.knowledge_file_active:
        return st.session_state.knowledge_file_object

    # Check if file exists locally before trying to upload
    if not os.path.exists(file_path):
         st.error(f"Knowledge file not found at: {file_path}. Please ensure it's in the correct location.")
         st.session_state.knowledge_file_active = False
         return None

    try:
        # Only upload if we haven't successfully uploaded before in this session
        if "knowledge_file_uri" not in st.session_state:
            st.info(f"Uploading knowledge file: {file_path}...")
            # display_name helps identify the file in your Google AI Studio
            knowledge_file = genai.upload_file(path=file_path, display_name="Accountant Knowledge Base MVP")
            st.session_state.knowledge_file_uri = knowledge_file.uri
            st.session_state.knowledge_file_object = knowledge_file # Store the object
            st.info("File uploaded. Waiting for processing...")
        else:
            # If URI exists but not active, retrieve the file object to check status
             knowledge_file = genai.get_file(name=st.session_state.knowledge_file_uri)
             st.session_state.knowledge_file_object = knowledge_file


        # Wait for the file to be processed (ACTIVE state) - Blocking operation
        wait_time = 1
        max_wait = 60 # Wait up to 60 seconds
        elapsed_wait = 0
        while st.session_state.knowledge_file_object.state.name == "PROCESSING" and elapsed_wait < max_wait:
            st.info(f"Processing uploaded file... (State: {st.session_state.knowledge_file_object.state.name}, waiting {wait_time}s)")
            time.sleep(wait_time)
            elapsed_wait += wait_time
            # Update file status
            st.session_state.knowledge_file_object = genai.get_file(st.session_state.knowledge_file_uri)
            wait_time = min(wait_time * 2, 10) # Exponential backoff, max 10s wait

        # Check final status
        if st.session_state.knowledge_file_object.state.name == "ACTIVE":
            st.session_state.knowledge_file_active = True
            st.success(f"Knowledge file '{st.session_state.knowledge_file_object.display_name}' is active and ready!")
            return st.session_state.knowledge_file_object
        else:
            st.error(f"File processing failed or timed out. Final State: {st.session_state.knowledge_file_object.state.name}")
            st.session_state.knowledge_file_active = False
            # Optionally delete the failed file
            # genai.delete_file(st.session_state.knowledge_file_uri)
            # del st.session_state.knowledge_file_uri
            return None

    except Exception as e:
        st.error(f"Error during knowledge file upload/processing: {e}")
        st.session_state.knowledge_file_active = False
        return None


def log_message_to_csv(timestamp, role, message):
    """Appends a message log entry to the CSV file."""
    file_exists = os.path.isfile(LOG_FILE)
    try:
        with open(LOG_FILE, 'a', newline='', encoding='utf-8') as csvfile:
            # Use csv.DictWriter for easier handling if columns change later
            fieldnames = ["timestamp", "role", "message"]
            log_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists or os.path.getsize(LOG_FILE) == 0:
                log_writer.writeheader()  # Write header if new/empty file

            log_writer.writerow({"timestamp": timestamp, "role": role, "message": message})
    except Exception as e:
        st.warning(f"Failed to log message to {LOG_FILE}: {e}")


# --- Streamlit App ---

st.title("📊 Chatbot4UXR")

# --- Initialization ---
# Initialize Gemini client (ensure API key is in secrets!)
model = setup_gemini_client()

# Attempt to upload/verify the knowledge file ONCE per session
# The function now uses session_state internally to manage this.
knowledge_file = upload_knowledge_file(KNOWLEDGE_CSV_PATH)

# Initialize chat history in session state using the structure from Streamlit tutorial
if "messages" not in st.session_state:
    st.session_state.messages = [] # List of {"role": "user/assistant", "content": ...}

# --- Chat History Display ---
# Display existing chat messages on rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input Handling ---
if prompt := st.chat_input("What can I help you with?"):
    # 1. Log and display user message
    timestamp_user = datetime.datetime.now().isoformat()
    # No need to log here if we log after potential modification/sending
    # log_message_to_csv(timestamp_user, "user", prompt) # Optional: log raw input immediately

    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Prepare and Call Gemini API ---
    # Check if knowledge file is ready before proceeding
    if not st.session_state.get("knowledge_file_active", False):
         st.warning("Knowledge file is not ready. Cannot generate response based on it.")
         # Decide if you want to proceed without the file or stop
         # Option: Stop
         st.stop()
         # Option: Proceed without file context (modify prompt below)
         # knowledge_file_for_prompt = None # Set to None if proceeding without file

    else:
         knowledge_file_for_prompt = st.session_state.knowledge_file_object

    # 2. Generate and stream assistant response
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # Use a placeholder for the streaming text
        full_response = "" # Accumulate response here if needed outside st.write_stream

        try:
            # Construct the history for the Gemini API
            # Convert Streamlit's format to Gemini's Content format
            history_for_gemini = []
            for msg in st.session_state.messages[:-1]: # Exclude the latest user prompt
                 role = "user" if msg["role"] == "user" else "model"
                 history_for_gemini.append({"role": role, "parts": [{"text": msg["content"]}]})

            # Start a chat session with history
            chat_session = model.start_chat(history=history_for_gemini)

            # Prepare the prompt parts for the current turn, including the file
            prompt_parts = []
            # Add file reference IF it's active and ready
            if knowledge_file_for_prompt:
                prompt_parts.append(knowledge_file_for_prompt)
                prompt_parts.append(f"\n\nStrictly using the provided file ('{knowledge_file_for_prompt.display_name}') and conversation history, answer the user's question: {prompt}")
            else: # If knowledge file isn't ready, send prompt without file context
                 st.warning("Proceeding without knowledge file context.")
                 prompt_parts.append(f"Using the conversation history, answer the user's question: {prompt}")


            # Send message with streaming enabled
            response_stream = chat_session.send_message(
                prompt_parts,
                stream=True
            )

            # Define a generator function to yield only the text parts from chunks
            def text_chunk_generator(stream):
                """Yields text from stream chunks, handling potential errors."""
                for chunk in stream:
                    try:
                        # Access the text part of the chunk
                        # Make sure chunk.text exists and handle potential errors if not
                        if hasattr(chunk, 'text'):
                             yield chunk.text
                        # You might need to inspect the exact chunk structure if text isn't directly available
                        # print(chunk) # Uncomment temporarily ONLY for debugging chunk structure
                    except Exception as e:
                         # Handle cases where a chunk might not have text or is malformed
                         # st.warning(f"Skipping chunk due to error: {e}") # Optional warning
                         pass # Silently skip chunks without text or causing errors

            # Use st.write_stream with the text-only generator
            full_response_text = st.write_stream(text_chunk_generator(response_stream))
            # --- !!! KEY CHANGE END !!! ---

        except Exception as e:
            error_message = f"An error occurred: {e}"
            st.error(error_message)
            full_response_text = error_message # Store error as the response for logging/history
            # Ensure the error is displayed if st.write_stream failed
            message_placeholder.markdown(error_message)


    # 3. Add assistant response (full text) to chat history and log it
    timestamp_assistant = datetime.datetime.now().isoformat()
    log_message_to_csv(timestamp_assistant, "assistant", full_response_text)
    # Make sure we only add the final text string to the session state
    st.session_state.messages.append({"role": "assistant", "content": full_response_text})

    # (Optional) Rerun if needed, though Streamlit usually handles reruns on widget interaction
    # st.rerun()