# 🧠 Semantic Visual Intelligence System (SVIS)

> A multimodal AI system combining Computer Vision and Generative AI for intelligent image understanding and interactive Visual Question Answering.

---

## 📋 Project Overview

This B.Tech Final Year Project implements a complete **Semantic Visual Intelligence System** that:

1. **Accepts an image** via a user-friendly Streamlit web interface
2. **Preprocesses** the image using PIL and NumPy pipelines
3. **Extracts visual features** using a pretrained BLIP Vision Transformer
4. **Generates a natural language caption** using the BLIP captioning model
5. **Refines the caption** using an LLM (OpenAI GPT or Groq LLaMA) for human-like quality
6. **Answers visual questions** via a BLIP-VQA model + LLM enhancement pipeline

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     Streamlit Web UI                        │
└──────────────────────────┬─────────────────────────────────┘
                           │ Image Upload
                           ▼
┌────────────────────────────────────────────────────────────┐
│               Image Preprocessing (PIL + NumPy)             │
│     • EXIF correction  • RGB normalization  • Resizing      │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│         Visual Feature Extraction (BLIP ViT Encoder)        │
│                   Salesforce/blip-image-captioning-base     │
└──────────┬────────────────────────────────────┬────────────┘
           │                                    │
           ▼                                    ▼
┌──────────────────────┐            ┌───────────────────────┐
│   Caption Generation  │            │     VQA Module         │
│   BLIP Captioning     │            │   BLIP-VQA + User Q    │
└──────────┬───────────┘            └──────────┬────────────┘
           │                                   │
           ▼                                   ▼
┌────────────────────────────────────────────────────────────┐
│              LLM Refinement Layer                           │
│    OpenAI GPT-3.5 / Groq LLaMA3 — Contextual Expansion    │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│              Natural Language Response to User              │
└────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
svis/
├── app.py                  # Streamlit entry point
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── README.md               # This file
│
├── app/
│   ├── __init__.py
│   └── ui.py               # All UI rendering logic
│
├── models/
│   ├── __init__.py
│   ├── captioner.py        # BLIP captioning + LLM refinement
│   └── vqa.py              # BLIP VQA + LLM answer expansion
│
├── utils/
│   ├── __init__.py
│   ├── image_utils.py      # Preprocessing, validation, stats
│   └── session.py          # Streamlit session management
│
├── static/
│   └── css/
│       └── style.css       # Custom UI styles
│
└── tests/
    ├── __init__.py
    └── test_core.py        # pytest unit tests
```

---

## ⚙️ Installation & Setup

### 1. Clone / Extract the project
```bash
cd svis
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** PyTorch with GPU support requires an extra step. Visit https://pytorch.org/get-started/locally/ for your platform-specific command.

### 4. Configure environment variables
```bash
cp .env.example .env
# Open .env and add your API key(s)
```

At least one of the following is recommended for **Advanced (LLM-Enhanced)** mode:
- `OPENAI_API_KEY` — from https://platform.openai.com/api-keys
- `GROQ_API_KEY` — free tier at https://console.groq.com/keys (LLaMA 3)

The system works **without any API key** in Standard mode — BLIP alone will generate captions.

### 5. Run the application
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 🧪 Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## 🤖 Models Used

| Task | Model | Source |
|------|-------|--------|
| Image Captioning | `Salesforce/blip-image-captioning-base` | Hugging Face |
| Visual QA | `Salesforce/blip-vqa-base` | Hugging Face |
| Caption Refinement | GPT-3.5 Turbo / LLaMA 3 (8B) | OpenAI / Groq |

Models are downloaded automatically on first run from Hugging Face (~900MB). Requires internet access on first launch.

---

## 💡 Usage Guide

1. Open the app at `http://localhost:8501`
2. Select **Standard** or **Advanced (LLM-Enhanced)** mode in the sidebar
3. Upload an image (JPG, PNG, WebP supported)
4. View the generated **caption** and **enhanced description**
5. Use the **VQA chat** to ask questions about the image
6. Use **Quick Question** buttons for common queries
7. Clear the chat anytime with the "Clear Chat" button

---

## 🌍 Real-World Applications

- **Assistive Technology** — Audio descriptions for visually impaired users
- **Smart Surveillance** — Automated scene understanding
- **Content Generation** — Alt-text and SEO descriptions for images
- **E-Learning** — Interactive visual education tools
- **Human-Computer Interaction** — Conversational visual interfaces

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** — Web interface
- **PyTorch + Transformers (Hugging Face)** — BLIP models
- **Pillow (PIL)** — Image handling
- **NumPy** — Numerical preprocessing
- **OpenAI / Groq SDK** — LLM refinement (optional)

---

## 📄 License

This project is developed for academic purposes as a B.Tech Final Year Project in Computer Science & Artificial Intelligence.
