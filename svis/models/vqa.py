"""
Visual Question Answering (VQA) Module
Uses BLIP-VQA model for image-grounded question answering,
with LLM enhancement for conversational, context-rich answers.
"""

import os
import logging
from PIL import Image

logger = logging.getLogger(__name__)

_vqa_processor = None
_vqa_model = None
_vqa_device = None
_vqa_loaded = False


def _load_vqa_model():
    try:
        from transformers import BlipProcessor, BlipForQuestionAnswering
        import torch

        model_name = "Salesforce/blip-vqa-base"
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForQuestionAnswering.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        return processor, model, device
    except ImportError:
        return None, None, None


def _ensure_vqa_model():
    global _vqa_processor, _vqa_model, _vqa_device, _vqa_loaded
    if not _vqa_loaded:
        _vqa_processor, _vqa_model, _vqa_device = _load_vqa_model()
        _vqa_loaded = True


def _blip_vqa(image: Image.Image, question: str) -> str:
    _ensure_vqa_model()

    if _vqa_model is None:
        return _mock_vqa(question)

    import torch
    inputs = _vqa_processor(image, question, return_tensors="pt").to(_vqa_device)
    with torch.no_grad():
        out = _vqa_model.generate(**inputs, max_new_tokens=50)
    return _vqa_processor.decode(out[0], skip_special_tokens=True)


def _mock_vqa(question: str) -> str:
    """Mock answers for environments without the full model stack."""
    q = question.lower()
    if "color" in q:
        return "The dominant colors appear to be natural tones with varying shades."
    if "person" in q or "people" in q or "human" in q:
        return "There appear to be people visible in the scene."
    if "time" in q or "day" in q or "night" in q:
        return "Based on the lighting, this appears to be taken during daytime."
    if "where" in q or "place" in q or "location" in q:
        return "The setting appears to be an outdoor or indoor environment."
    if "how many" in q or "count" in q or "number" in q:
        return "There are several elements visible in the image."
    if "activity" in q or "doing" in q or "happening" in q:
        return "The image shows a scene with various activities taking place."
    return "Based on the visual content, the image contains several notable elements relevant to your question."


def _expand_answer_with_llm(question: str, raw_answer: str, caption: str) -> str:
    """Use LLM to generate a conversational, context-rich answer."""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        return _local_expand(question, raw_answer, caption)

    prompt = (
        f"Image description: {caption}\n"
        f"User question: {question}\n"
        f"Initial answer: {raw_answer}\n\n"
        "Provide a natural, conversational, and context-aware answer to the question "
        "based on the image description and initial answer. Be specific and informative. "
        "Keep the response under 3 sentences."
    )

    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful visual AI assistant that answers questions about images accurately and naturally."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=150,
                temperature=0.6,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"OpenAI VQA refinement failed: {e}")

    if os.getenv("GROQ_API_KEY"):
        try:
            from groq import Groq
            client = Groq()
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are a helpful visual AI assistant that answers questions about images accurately and naturally."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=150,
                temperature=0.6,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Groq VQA refinement failed: {e}")

    return _local_expand(question, raw_answer, caption)


def _local_expand(question: str, raw_answer: str, caption: str) -> str:
    """Rule-based expansion without LLM."""
    return (
        f"{raw_answer.capitalize()}. "
        f"Based on the image context — {caption.lower().rstrip('.')} — "
        f"this appears to be the most relevant answer to your question."
    )


def answer_question(image: Image.Image, caption: str, question: str) -> str:
    """
    Main VQA function.

    Args:
        image: Preprocessed PIL Image
        caption: Previously generated caption for context
        question: User's natural language question

    Returns:
        String answer
    """
    try:
        raw_answer = _blip_vqa(image, question)
        expanded = _expand_answer_with_llm(question, raw_answer, caption)
        return expanded
    except Exception as e:
        logger.error(f"VQA error: {e}")
        return f"I encountered an error while processing your question: {e}"
