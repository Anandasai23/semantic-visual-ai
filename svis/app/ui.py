"""
UI rendering module for the Semantic Visual Intelligence System
"""

import streamlit as st
from PIL import Image
import io

from models.captioner import generate_caption
from models.vqa import answer_question
from utils.image_utils import preprocess_image, validate_image
from utils.session import init_session_state


def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class='sidebar-brand'>
            <span class='brand-icon'>🧠</span>
            <span class='brand-text'>SVIS</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### ⚙️ Settings")

        mode = st.selectbox(
            "Model Mode",
            ["Standard (BLIP)", "Advanced (LLM-Enhanced)"],
            index=1,
            help="Standard uses BLIP only. Advanced refines captions with an LLM."
        )

        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown("""
        **Semantic Visual Intelligence System** combines Computer Vision and Generative AI to:
        - Generate natural language image descriptions
        - Answer questions about uploaded images
        - Provide context-aware visual analysis

        **Pipeline:**
        1. Image Upload & Preprocessing
        2. Visual Feature Extraction
        3. Caption Generation (BLIP)
        4. LLM Refinement
        5. Interactive VQA
        """)

        st.markdown("---")
        st.caption("B.Tech Final Year Project · CS & AI")

    return mode


def render_upload_section():
    st.markdown("<h2 class='section-header'>📤 Upload an Image</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Supported formats: JPG, PNG, BMP, WebP"
        )

    with col2:
        st.markdown("""
        <div class='tip-box'>
            <b>💡 Tips for best results:</b><br>
            • Clear, well-lit images<br>
            • Resolution ≥ 224×224px<br>
            • Avoid heavily blurred images<br>
            • Works with photos, diagrams, screenshots
        </div>
        """, unsafe_allow_html=True)

    return uploaded_file


def render_image_preview(image: Image.Image):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    w, h = image.size
    mode = image.mode
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Width", f"{w}px")
    col_b.metric("Height", f"{h}px")
    col_c.metric("Color Mode", mode)


def render_caption_section(image: Image.Image, llm_mode: bool):
    st.markdown("<h2 class='section-header'>🗣️ Generated Caption</h2>", unsafe_allow_html=True)

    with st.spinner("Analyzing image and generating caption..."):
        processed = preprocess_image(image)
        result = generate_caption(processed, use_llm=llm_mode)

    if result["status"] == "success":
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"""
            <div class='caption-box'>
                <span class='caption-label'>Caption</span>
                <p class='caption-text'>{result['caption']}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class='meta-box'>
                <b>Model Used</b><br><span class='meta-value'>{result.get('model','BLIP')}</span><br><br>
                <b>Confidence</b><br><span class='meta-value'>{result.get('confidence','–')}</span>
            </div>
            """, unsafe_allow_html=True)

        if llm_mode and result.get("refined_caption"):
            st.markdown("<h3 class='subsection-header'>✨ LLM-Enhanced Description</h3>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='refined-box'>
                {result['refined_caption']}
            </div>
            """, unsafe_allow_html=True)

        return result["caption"]
    else:
        st.error(f"Caption generation failed: {result.get('error', 'Unknown error')}")
        return None


def render_vqa_section(image: Image.Image, caption: str):
    st.markdown("<h2 class='section-header'>💬 Visual Question Answering</h2>", unsafe_allow_html=True)
    st.markdown("Ask anything about the image — objects, activities, environment, context, and more.")

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Render existing messages
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class='chat-msg user-msg'>
                    <span class='chat-avatar user-avatar'>You</span>
                    <div class='chat-bubble user-bubble'>{msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='chat-msg ai-msg'>
                    <span class='chat-avatar ai-avatar'>AI</span>
                    <div class='chat-bubble ai-bubble'>{msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)

    # Quick question suggestions
    st.markdown("**Quick Questions:**")
    suggestions = [
        "What objects are in this image?",
        "Describe the setting or environment.",
        "What activity is happening?",
        "What time of day does this appear to be?",
        "Are there any people in this image?",
    ]
    cols = st.columns(len(suggestions))
    for i, (col, s) in enumerate(zip(cols, suggestions)):
        with col:
            if st.button(s, key=f"sugg_{i}", use_container_width=True):
                st.session_state.pending_question = s

    # Question input
    question = st.text_input(
        "Ask a question about the image",
        value=st.session_state.get("pending_question", ""),
        placeholder="e.g. What is the main subject of this image?",
        key="vqa_input"
    )

    if "pending_question" in st.session_state:
        del st.session_state.pending_question

    col_ask, col_clear = st.columns([3, 1])
    with col_ask:
        ask_clicked = st.button("🔍 Ask", type="primary", use_container_width=True)
    with col_clear:
        clear_clicked = st.button("🗑 Clear Chat", use_container_width=True)

    if clear_clicked:
        st.session_state.chat_history = []
        st.rerun()

    if ask_clicked and question.strip():
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            processed = preprocess_image(image)
            answer = answer_question(processed, caption, question)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()


def render_main():
    init_session_state()

    # Header
    st.markdown("""
    <div class='app-header'>
        <h1 class='app-title'>🧠 Semantic Visual Intelligence System</h1>
        <p class='app-subtitle'>Multimodal AI · Computer Vision + Generative AI · Image Understanding & Visual QA</p>
    </div>
    """, unsafe_allow_html=True)

    mode = render_sidebar()
    llm_mode = "Advanced" in mode

    st.markdown("---")

    # Upload
    uploaded_file = render_upload_section()

    if uploaded_file:
        st.session_state.current_image_name = uploaded_file.name

        try:
            image = Image.open(uploaded_file).convert("RGB")
            valid, msg = validate_image(image)
            if not valid:
                st.warning(f"⚠️ {msg}")
                return
        except Exception as e:
            st.error(f"Failed to load image: {e}")
            return

        st.markdown("---")
        render_image_preview(image)

        st.markdown("---")
        caption = render_caption_section(image, llm_mode)

        if caption:
            st.markdown("---")
            render_vqa_section(image, caption)

    else:
        # Landing placeholder
        st.markdown("""
        <div class='landing-placeholder'>
            <div class='placeholder-icon'>🖼️</div>
            <h3>Upload an image to get started</h3>
            <p>The system will generate a natural language caption and allow you to ask questions about the image.</p>
        </div>
        """, unsafe_allow_html=True)
