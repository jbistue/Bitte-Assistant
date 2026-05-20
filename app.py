import os
import time
import streamlit as st
import base64
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Any

load_dotenv()

st.set_page_config(page_title="Bitte RAG ChatBot", page_icon=":material/chat_bubble:", layout="centered")

st.title("🤖 Bitte RAG ChatBot")
st.markdown("**Your intelligent assistant powered by GPT-5 and RAG technology**")
st.divider()

st.expander("ℹ️ About this webapp", expanded=False).markdown(
    """
    **Bitte RAG ChatBot**

    Model: `GPT-5` via OpenAI Responses API  
    RAG: File Search tool using your pre-built Vector Store  
    Features: multi-turn chat, image inputs, clear conversation  
    Secrets: reads your `OPENAI_API_KEY` and `VECTOR_STORE_ID` from Streamlit secrets or environment variables\n
    
    **How it works**  
    Your message and (optional) images go to the Responses API along with a system prompt  
    The File Search tool retrieves relevant passages from your Vector Store to ground the answer  
    Click the "Ask Bitte" button to get a response
    """
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID") or st.secrets.get("VECTOR_STORE_ID")

client = OpenAI(api_key=OPENAI_API_KEY)

if not OPENAI_API_KEY:
    st.error("Please set your OPENAI_API_KEY in Streamlit secrets or environment variables.")

if  not VECTOR_STORE_ID:
    st.error("Please set your VECTOR_STORE_ID in Streamlit secrets or environment variables.")

# Configuration of the system prompt
SYSTEM_PROMPT = """
You are a toxic CEO who loves things like pre-revenue or cash burn ratio.
"""

# Store the previous response id
if "previous_response_id" not in st.session_state:
    st.session_state.previous_response_id = None

# Initialize the chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create a sidebar with user controls
with st.sidebar:
    st.header("User Controls")
    st.divider()
    # Clear the conversation history
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.previous_response_id = None
        st.success("Conversation history cleared.")
        time.sleep(2)
        # Reset the page
        st.rerun()
    st.divider()
    st.markdown("**Note:** Clearing the conversation will reset the context for the chatbot.")


# Helper functions
def build_input_parts(prompt: str, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build the input parts array for the OpenAI from text and images

    :param str prompt: text prompt from the user
    :param List[Dict[str, Any]] images: list of images with their mime types and data URLs
    :param str previous_response_id: ID of the previous response for context
    :return List[Dict[str, Any]]: list of input parts for the OpenAI API
    """
    content = []
    if prompt and prompt.strip():
        content.append({"type": "input_text", "text": prompt.strip()})
    for image in images:
        content.append({
            "type": "input_image",
            "image_url": image["data_url"],
            "detail": "auto",
        })
    return [{"type": "message", "role": "user", "content": content}] if content else []


# Function to generate a response from the OpenAI API
def call_responses_api(input_parts: List[Dict[str, Any]], previous_response_id: str = None) -> Any:
    """Generate a response from the OpenAI API based on the input parts

    :param List[Dict[str, Any]] input_parts: list of input parts for the OpenAI API
    :param str previous_response_id: ID of the previous response for context
    :return Any: generated response
    """
    # if not input_parts:
    #     return "Please enter a message or upload an image to get a response."

    # Build the messages array for the API call
    # messages = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages + input_parts

    response = client.responses.create(
        model="gpt-5-nano",
        input=input_parts,
        instructions=SYSTEM_PROMPT,
        tools=[{
            "type": "file_search",
            "vector_store_ids": [VECTOR_STORE_ID],
            "max_num_results": 20,
        }],
        previous_response_id=previous_response_id,
        # max_tokens=500,
    )

    return response


# Function to get the text output
def get_text_output(response: Any) -> str:
    """Extract the text output from the API response

    :param Any response: API response object
    :return str: extracted text output
    """
    
    # if not response or not response.choices:
    #     return "No response generated. Please try again."

    # Extract the text content from the response parts
    # text_output = ""
    # for part in response.choices[0].message.content:
    #     if part["type"] == "text":
    #         text_output += part["text"] + "\n"

    # return text_output.strip()
    
    return response.output_text
    

# Render all previous messages in the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            for input_part in message["content"]:
                if input_part["type"] == "message":
                    for content_part in input_part["content"]:
                        if content_part["type"] == "input_text":
                            st.markdown(content_part["text"])
                        elif content_part["type"] == "input_image":
                            st.image(content_part["image_url"], width=100)
        else:
            st.markdown(message["content"]) 

# User interface - upload images
uploaded_files = st.file_uploader("Upload images (optional)",
                                   type=["png", "jpg", "jpeg", "webp"], 
                                   accept_multiple_files=True,
                                   key=f"file_uploader_{len(st.session_state.messages)}")  # Unique key to reset uploader after each message

# User interface - chat input
prompt = st.chat_input("Type your message here...")

if prompt is not None:
    # Process only the currently uploaded images into an API-compatible format
    images = []
    if uploaded_files:
        images = [
            {
                "mime_type": f"image/{file.type.split('/')[-1]}" if file.type else "image/png",
                "data_url": f"data:{file.type};base64,{base64.b64encode(file.read()).decode('utf-8')}",
            }
            for file in uploaded_files
        ]

    # Build the input parts for the responses API
    input_parts = build_input_parts(prompt, images)

    # Store the messages
    st.session_state.messages.append({"role": "user", "content": input_parts})

    # Display the user's message in the chat
    with st.chat_message("user"):
        for input_part in input_parts:
            if input_part["type"] == "message":
                for content_part in input_part["content"]:
                    if content_part["type"] == "input_text":
                        st.markdown(content_part["text"])
                    elif content_part["type"] == "input_image":
                        st.image(content_part["image_url"], width=100)

    # Generate a response from the OpenAI API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = call_responses_api(input_parts, st.session_state.previous_response_id)
                output_text = get_text_output(response)

                # Display the AI's response in the chat
                st.markdown(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})
                # Retrieve the response ID for context in the next turn if available
                st.session_state.previous_response_id = response.id # if response else None
            except Exception as e:
                st.error(f"Error generating response: {e}")
