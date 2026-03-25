# 🧠 Semantic Visual Intelligence System (SVIS)

> A multimodal AI system combining Computer Vision and Generative AI for intelligent image understanding and interactive Visual Question Answering.

🔗 **GitHub:** https://github.com/Anandasai23/semantic-visual-ai

---

## 📋 Project Description

Semantic Visual Intelligence System (SVIS) is a multimodal artificial intelligence application that integrates Computer Vision and Generative AI to enable intelligent image understanding and interactive human-machine communication. The system accepts an image as input through a web-based interface and processes it through a sequential AI pipeline to produce natural language descriptions and context-aware answers to user queries.

The core pipeline begins with image acquisition and preprocessing using PIL and NumPy, followed by deep visual feature extraction using **BLIP (Bootstrapped Language-Image Pretraining)**, a state-of-the-art vision-language transformer model developed by Salesforce. The extracted visual features are passed through a captioning decoder that generates an initial natural language description of the image content.

To elevate the quality of this output, the system integrates a **Large Language Model (LLM)** — either Groq's LLaMA 3 or OpenAI's GPT-3.5 — which refines the raw caption into a rich, human-like, and context-aware description. Beyond captioning, SVIS supports **Visual Question Answering (VQA)**, where users can ask open-ended natural language questions about the uploaded image and receive accurate, conversational responses.

---

## 🏗️ System Architecture
```
Input Image → Preprocessing → BLIP Feature Extraction
     ↓                                  ↓
Streamlit UI              Caption Generation + VQA
     ↓                                  ↓
User Query  →        LLM Refinement Layer (Groq / OpenAI)
                                        ↓
                          Natural Language Response
```

---

## 📁 Project Structure
```
svis/
├── app.py                      # Streamlit entry point
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── README.md                   # This file
├── run.bat                     # One-click launcher (Windows)
├── run.sh                      # One-click launcher (Mac/Linux)
├── pytest.ini                  # Test configuration
├── .streamlit/
│   └── config.toml             # Streamlit theme and server config
├── app/
│   ├── __init__.py
│   └── ui.py                   # All UI rendering logic
├── models/
│   ├── __init__.py
│   ├── captioner.py            # BLIP captioning + LLM refinement
│   └── vqa.py                  # BLIP VQA + LLM answer expansion
├── utils/
│   ├── __init__.py
│   ├── image_utils.py          # Preprocessing, validation, stats
│   └── session.py              # Streamlit session management
├── static/
│   └── css/
│       └── style.css           # Custom UI styles
└── tests/
    ├── __init__.py
    └── test_core.py            # pytest unit tests
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+ → https://python.org/downloads
- VS Code → https://code.visualstudio.com
- Git → https://git-scm.com/download/win

### Step 1 — Clone the Repository
```bash
git clone https://github.com/Anandasai23/semantic-visual-ai.git
cd semantic-visual-ai/svis
```

### Step 2 — Create Virtual Environment
```bash
python -m venv venv
```

### Step 3 — Activate Virtual Environment

**Windows:**
```bash
.\venv\Scripts\Activate.ps1
```

If you get a permissions error, run this once first:
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### Step 4 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5 — Configure API Key
```bash
copy .env.example .env
```

Open `.env` and add your free Groq API key from https://console.groq.com/keys:
```
GROQ_API_KEY=gsk_your_key_here
```

### Step 6 — Run the App
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser ✅

---

## 🔁 Running After Reopening VS Code

Every time you reopen VS Code, run these 3 commands:
```bash
cd svis
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

### One-Click Desktop Shortcut (Windows)

Create `Launch_SVIS.bat` on your Desktop:
```bat
@echo off
cd D:\Claude\Projects\svis_project\svis
call venv\Scripts\activate
streamlit run app.py
pause
```

---

## 💡 Usage Guide

1. Open **http://localhost:8501** in your browser
2. Select **Standard** or **Advanced (LLM-Enhanced)** mode in the sidebar
3. Upload any image — JPG, PNG, BMP, or WebP
4. Wait for caption generation (first run downloads models — 2 to 5 minutes)
5. View the generated **caption** and **LLM-enhanced description**
6. Use the **VQA chat** to ask questions about the image
7. Use **Quick Question** buttons for common queries
8. Click **Clear Chat** to reset the conversation

---

## 🤖 Models Used

| Task | Model | Source |
|------|-------|--------|
| Image Captioning | Salesforce/blip-image-captioning-base | Hugging Face |
| Visual Question Answering | Salesforce/blip-vqa-base | Hugging Face |
| LLM Refinement | LLaMA 3 8B via Groq | Groq API |
| LLM Refinement (alt) | GPT-3.5 Turbo | OpenAI API |

> Models download automatically on first run (~900 MB). Internet required on first launch only.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.10+ | Core programming language |
| Streamlit | Web interface |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | BLIP vision-language models |
| Pillow (PIL) | Image loading and preprocessing |
| NumPy | Numerical operations |
| Groq SDK | LLaMA 3 LLM integration |
| OpenAI SDK | GPT-3.5 integration (optional) |

---

## 🌍 Real-World Applications

| Domain | Use Case |
|--------|----------|
| Assistive Technology | Audio descriptions for visually impaired users |
| Smart Surveillance | Automated scene understanding and monitoring |
| Content Generation | Alt-text and SEO descriptions for images |
| E-Learning | Interactive visual education tools |
| Human-Computer Interaction | Conversational visual interfaces |

---

## 🧪 Running Tests
```bash
pip install pytest
pytest tests/ -v
```

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| `streamlit not recognized` | Activate venv first and make sure you are inside the `svis` folder |
| `No such file: requirements.txt` | Run `cd svis` before installing |
| `Activate.ps1 error` | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| App slow on first image | Normal — BLIP models downloading (~900 MB), wait 2 to 5 minutes |
| LLM refinement not working | Add `GROQ_API_KEY` to your `.env` file |
| GitHub push blocked | Never commit `.env` or real API keys |

---

## 🔐 Security Notes

- Never put real API keys in `.env.example` — keep it as a blank template
- Your `.env` file is gitignored and will never be pushed to GitHub
- If you accidentally expose a key, revoke it immediately at https://console.groq.com/keys

---

## 👨‍💻 Developer

**Anandasai** · GitHub: [@Anandasai23](https://github.com/Anandasai23)

---

## 📄 License

This project is developed for academic purposes as a B.Tech Final Year Project in Computer Science & Artificial Intelligence.
