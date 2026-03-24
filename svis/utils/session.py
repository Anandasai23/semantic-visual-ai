"""
Streamlit session state management utilities.
"""

import streamlit as st


def init_session_state():
    """Initialize all required session state variables."""
    defaults = {
        "chat_history": [],
        "current_caption": None,
        "current_image_name": None,
        "pending_question": None,
        "model_loaded": False,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default
