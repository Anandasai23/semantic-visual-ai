"""
Semantic Visual Intelligence System
Main Streamlit Application Entry Point
"""

import streamlit as st

st.set_page_config(
    page_title="Semantic Visual Intelligence System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
with open("static/css/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from app.ui import render_main

if __name__ == "__main__":
    render_main()
