"""
Ajax Tactiek Bot - A Streamlit application for answering questions about Ajax football tactics.

This module provides a web interface for users to ask questions about Ajax football tactics
and receive informative responses using a combination of vector search and language models.

The application features:
- A modern, Ajax-themed UI with responsive design
- Real-time question processing with loading animations
- Session state management for maintaining conversation context
- Error handling for API key validation and response generation
"""

import streamlit as st
import time
import random
from typing import Optional
from dotenv import load_dotenv
import os
import sys
from pathlib import Path

# Add src directory to Python path for imports
src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir))

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from llm.get_question_answer import get_football_answer

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def validate_api_key() -> None:
    """
    Validate the presence of the OpenAI API key.
    
    Raises:
        SystemExit: If the API key is not found in environment variables.
    """
    if not OPENAI_API_KEY:
        st.error("OpenAI API key not found. Please add it to your .env file.")
        st.stop()

def configure_page() -> None:
    """
    Configure the Streamlit page settings.
    
    Sets up the page title, icon, and layout for optimal user experience.
    """
    st.set_page_config(
        page_title="Ajax Tactiek Bot",
        page_icon="⚽",
        layout="centered"
    )

def load_css() -> None:
    """
    Load custom CSS to style the application with Ajax theme.
    
    Defines styles for:
    - Main container and background
    - Header with logo and title
    - Question input card
    - Response display
    - Loading animations
    - Buttons and interactive elements
    """
    st.markdown(
        """
    <style>
        /* Main container styling */
        .stApp {
            background: linear-gradient(to bottom, white, #f3f4f6);
        }
        
        /* Header styling */
        .header-container {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1.5rem;
        }
        
        .header-logo {
            height: 80px;
        }
        
        .header-title {
            font-size: 2rem;
            font-weight: bold;
            color: #ea384c;
            margin-left: 1rem;
        }
        
        /* Card styling */
        .card {
            border: 2px solid rgba(234, 56, 76, 0.2);
            border-radius: 0.5rem;
            background-color: white;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        
        .card-header {
            background-color: #ea384c;
            color: white;
            padding: 1.5rem;
            border-top-left-radius: 0.3rem;
            border-top-right-radius: 0.3rem;
            text-align: center;
        }
        
        .card-title {
            font-size: 1.5rem;
            font-weight: 600;
        }
        
        .card-content {
            padding: 1.5rem 1rem;
        }
        
        /* Response card styling */
        .response-card {
            border-left: 4px solid #ea384c;
            padding: 1rem;
            background-color: white;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
            margin-top: 1.5rem;
            animation: fadeIn 0.5s;
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            font-size: 0.875rem;
            color: #6b7280;
            margin-top: 1.5rem;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #ea384c;
            color: white;
            border: none;
            padding: 0.5rem 2rem;
            font-weight: 500;
            border-radius: 0.25rem;
            transition: background-color 0.2s;
        }
        
        .stButton > button:hover {
            background-color: #d1293c;
        }
        
        .stButton > button:disabled {
            background-color: #ea384c;
            opacity: 0.7;
        }
        
        /* Loading animation */
        .loading-animation {
            display: flex;
            justify-content: center;
            margin: 2rem 0;
        }
        
        .loading-spinner {
            width: 4rem;
            height: 4rem;
            border-radius: 50%;
            border: 4px solid rgba(234, 56, 76, 0.2);
            border-top-color: #ea384c;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            to {
                transform: rotate(360deg);
            }
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

def display_header() -> None:
    """
    Display the application header with Ajax logo and title.
    
    Creates a centered header with the Ajax logo and application title,
    styled according to the Ajax brand colors.
    """
    st.markdown(
        """
    <div class="header-container">
        <img src="https://upload.wikimedia.org/wikipedia/en/7/79/Ajax_Amsterdam.svg" class="header-logo" alt="Ajax Logo">
        <h1 class="header-title">Ajax Tactiek Bot</h1>
    </div>
    """,
        unsafe_allow_html=True,
    )

def display_question_form() -> Optional[str]:
    """
    Display the question input form and process submissions.
    
    Returns:
        Optional[str]: The submitted question if the form was submitted, None otherwise.
    """
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-header"><h2 class="card-title">Wat wil je weten over de tactiek van Ajax?</h2></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="card-content">', unsafe_allow_html=True)

    with st.form("question_form"):
        question = st.text_area(
            "Vraag",
            value="",
            placeholder="Bijv. Hoe speelt Ajax met de vleugelspelers?",
            height=100,
            key="question_input",
        )

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submit_button = st.form_submit_button("Vraag Stellen")

        if submit_button and question.strip():
            st.session_state.is_loading = True
            st.session_state.response = ""
            return question.strip()

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    return None

def display_loading_animation() -> None:
    """
    Display a loading animation while processing the question.
    
    Shows a spinning animation in the Ajax brand colors to indicate
    that the system is processing the user's question.
    """
    st.markdown(
        """
    <div class="loading-animation">
        <div class="loading-spinner"></div>
    </div>
    """,
        unsafe_allow_html=True,
    )

def display_response() -> None:
    """
    Display the response from the football tactics bot.
    
    Shows the generated response in a styled card with a fade-in animation
    if a response exists in the session state.
    """
    if st.session_state.response:
        st.markdown(
            f"""
        <div class="response-card">
            <div>{st.session_state.response}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

def display_footer() -> None:
    """
    Display the application footer.
    
    Shows copyright information and the year in a centered footer.
    """
    st.markdown(
        '<div class="footer"><p>Ontwikkeld voor Ajax fans • 2023</p></div>',
        unsafe_allow_html=True,
    )

def initialize_session_state() -> None:
    """
    Initialize session state variables.
    
    Sets up the necessary session state variables for managing
    the application's state between reruns.
    """
    if "response" not in st.session_state:
        st.session_state.response = ""
    if "is_loading" not in st.session_state:
        st.session_state.is_loading = False

def process_question(question: str) -> None:
    """
    Process the user's question and generate a response.
    
    Args:
        question (str): The user's question about Ajax tactics.
    """
    try:
        # Initialize FAISS database
        embeddings = OpenAIEmbeddings()
        db_path = Path(__file__).resolve().parent.parent.parent / "data"
        faiss_db = FAISS.load_local(str(db_path), embeddings, allow_dangerous_deserialization = True)
        
        # Get answer from the football QA system
        response = get_football_answer(faiss_db, question)
        
        # Update session state with the response
        st.session_state.response = response
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        st.session_state.response = "Sorry, er is een fout opgetreden bij het verwerken van je vraag. Probeer het later opnieuw."
    finally:
        # Ensure loading state is set to False
        st.session_state.is_loading = False
        # Force a rerun to update the UI
        st.rerun()

def main() -> None:
    """
    Main application entry point.
    
    Orchestrates the flow of the application by:
    1. Validating the API key
    2. Configuring the page
    3. Loading styles
    4. Initializing session state
    5. Displaying UI components
    6. Processing user input
    """
    validate_api_key()
    configure_page()
    load_css()
    initialize_session_state()
    
    display_header()
    question = display_question_form()
    
    # Show loading animation if we're in loading state
    if st.session_state.is_loading:
        display_loading_animation()
    
    # Process the question if we have one
    if question:
        process_question(question)
    
    # Always display response if we have one
    display_response()
    display_footer()

if __name__ == "__main__":
    main()
