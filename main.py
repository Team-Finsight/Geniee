import streamlit as st
from streamlit_chat import message
import requests
from dataai import display_data_ai_session
import subprocess
import requests
import tempfile
import time
import os
from doc_chat import allowed_file, extract_text_from_docx, extract_text_from_image, pdf_to_text, ocr_pdf_to_text, qa
import pandas as pd
from general_chat import chat


def extract_text(file_buffer, file_name):
    """Extract text from uploaded file buffer based on its type."""
    try:
        if allowed_file(file_name, {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'docx', 'doc'}):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as temp_file:
                temp_file.write(file_buffer.read())
                temp_file.flush()
                if temp_file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    text = extract_text_from_image(temp_file.name)
                elif temp_file.name.lower().endswith('.pdf'):
                    text = ocr_pdf_to_text(temp_file.name)
                elif temp_file.name.lower().endswith(('.docx', '.doc')):
                    text = extract_text_from_docx(temp_file.name)
                else:
                    os.unlink(temp_file.name)  # Delete the temporary file
                    return "error: Unsupported file format"
    except Exception as e:
        return str(e)
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)  # Ensure the temporary file is deleted
    return text
def display_chat_ui(chat, chat_function):
    """ Generic function to display chat UI. """
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='chat_form', clear_on_submit=True):
            col1, col2 = st.columns([0.9, 0.1])
            with col1:
                # Check if there's a last_message to pre-populate, else default placeholder
                last_message = st.session_state.get('last_message', "")
                user_input = st.text_area("Message:", value=last_message, placeholder="Type your message here", key='chat_input', height=100)
            with col2:
                regenerate_button = st.form_submit_button(label='🔄')

            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                response = chat(user_input)
                print(response)
                if response:
                    chat_function(user_input, response)
            # Clear the last_message after submitting
            st.session_state['last_message'] = ""

        if regenerate_button:
            # If there's a past message to regenerate, do not clear the input but trigger the response generation
            if 'past' in st.session_state and st.session_state['past']:
                # Get the last user message
                last_message = st.session_state['past'][-1]
                # Set last_message to regenerate the form with it
                st.session_state['last_message'] = last_message
                # Optionally, you can call the function directly here if you want to process it immediately
                response = chat(last_message)
                if response:
                    chat_function(last_message, response)

    if 'generated' in st.session_state and st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

def update_chat_history(user_input,response):
    """ Update chat history with the response from the API. """
    # Check if the 'response' key is in the API response and update the session state accordingly\
    if response:
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(response)
    else:
        st.error("Received an unexpected response format from the API.")


def display_chat_with_documents():
    """ Display chat interface for document-based conversations. """
    file_buffer = st.sidebar.file_uploader("Upload a file", type=['txt', 'pdf', 'png', 'jpg', 'jpeg', 'docx', 'doc'])
    if file_buffer is not None:
        text = ""
        for file in file_buffer:
            try:
                file_name = file_buffer.name
                file_content = extract_text(file_buffer, file_name)
                # Assuming extract_text is implemented to handle different file types
                
                text += file_content 
                #print(text) # Append file content and name, then add a newline for separation
            except Exception as e:
                st.error(f"{str(e)}")

        # Container for chat messages
        reply_container = st.container()
        container = st.container()

        with container:
            with st.form(key='chat_form', clear_on_submit=True):
                col1, col2 = st.columns([0.9, 0.1])
                with col1:
                    # Check if there's a last_message to pre-populate, else default placeholder
                    last_message = st.session_state.get('last_message', "")
                    user_input = st.text_area("Message:", value=last_message, placeholder="Type your message here", key='chat_input', height=100)
                with col2:
                    regenerate_button = st.form_submit_button(label='🔄')

                submit_button = st.form_submit_button(label='Send')

                if submit_button and user_input:
                    with st.spinner('Generating response...'):
                        # Call the function to handle the conversation, passing the selected document ID and the user query
                        output = qa(text, user_input)
                        if output:
                            # Extract the actual response from the output if necessary
                            response_message = output
                            # Update session state with the new messages
                            update_chat_history(user_input, response_message)

                if regenerate_button:
                    # If there's a past message to regenerate, trigger the response generation again
                    if 'past' in st.session_state and st.session_state['past']:
                        # Get the last user message
                        last_message = st.session_state['past'][-1]
                        # Optionally, you can trigger response generation directly here
                        output = qa(text, last_message)
                        if output:
                            response_message = output
                            update_chat_history(last_message, response_message)

        # Display chat history using session state
        if 'generated' in st.session_state and st.session_state['generated']:
            with reply_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")


def initialize_session_state():
    """ Initialize session state. """
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Welcome to Geniee!"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hello!"]

def main():
    initialize_session_state()
    st.title("Geniee🫡")
    chat_mode = st.sidebar.radio("Choose your interaction mode:", ['General Chat', 'Chat with Documents', 'Data-AI'])

    if chat_mode == 'General Chat':
        display_chat_ui(chat, update_chat_history)
    elif chat_mode == 'Chat with Documents':
        display_chat_with_documents()
    elif chat_mode == 'Data-AI':
        display_data_ai_session()

if __name__ == "__main__":
    main()
