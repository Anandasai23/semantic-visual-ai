🧠 Semantic Visual Intelligence System (SVIS)
A multimodal AI system combining Computer Vision and Generative AI for intelligent image understanding and interactive Visual Question Answering.
📋 Project Description
Semantic Visual Intelligence System (SVIS) is a multimodal artificial intelligence application that integrates Computer Vision and Generative AI to enable intelligent image understanding and interactive human-machine communication. The system accepts an image as input through a web-based interface and processes it through a sequential AI pipeline to produce natural language descriptions and context-aware answers to user queries.
The core pipeline begins with image acquisition and preprocessing using PIL and NumPy, followed by deep visual feature extraction using BLIP (Bootstrapped Language-Image Pretraining), a state-of-the-art vision-language transformer model developed by Salesforce. The extracted visual features are passed through a captioning decoder that generates an initial natural language description of the image content.
To elevate the quality of this output, the system integrates a Large Language Model (LLM) — either Groq's LLaMA 3 or OpenAI's GPT-3.5 — which refines the raw caption into a rich, human-like, and context-aware description. Beyond captioning, SVIS supports Visual Question Answering (VQA), where users can ask open-ended natural language questions about the uploaded image and receive accurate, conversational responses.
🏗️ System Architecture:
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
│              Salesforce/blip-image-captioning-base          │
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
│                   LLM Refinement Layer                      │
│    Groq LLaMA3 / OpenAI GPT-3.5 — Contextual Expansion    │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
📁 Project Structure:
┌───────────────────────────svis/
├── app.py                      # Streamlit entry point
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── README.md                   # This file
├── run.bat                     # One-click launcher (Windows)
├── run.sh                      # One-click launcher (Mac/Linux)
├── pytest.ini                  # Test configuration
│
├── .streamlit/
│   └── config.toml             # Streamlit theme and server config
│
├── app/
│   ├── __init__.py
│   └── ui.py                   # All UI rendering logic
│
├── models/
│   ├── __init__.py
│   ├── captioner.py            # BLIP captioning + LLM refinement
│   └── vqa.py                  # BLIP VQA + LLM answer expansion
│
├── utils/
│   ├── __init__.py
│   ├── image_utils.py          # Preprocessing, validation, stats
│   └── session.py              # Streamlit session management
│
├── static/
│   └── css/
│       └── style.css           # Custom UI styles
│
└── tests/
    ├── __init__.py
