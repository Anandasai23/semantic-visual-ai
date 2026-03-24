"""
Image Captioning Module
Uses BLIP (Bootstrapped Language-Image Pretraining) for caption generation,
with optional LLM refinement via Groq or OpenAI API.
"""

import os
import logging
from PIL import Image

logger = logging.getLogger(__name__)


def _load_blip_model():
    """Lazy-load BLIP model to avoid startup delay."""
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch

        model_name = "Salesforce/blip-image-captioning-base"
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        return processor, model, device
    except ImportError:
        logger.warning("Transformers not installed. Using mock captioner.")
        return None, None, None


_processor = None
_model = None
_device = None
_model_loaded = False


def _ensure_model():
    global _processor, _model, _device, _model_loaded
    if not _model_loaded:
        _processor, _model, _device = _load_blip_model()
        _model_loaded = True


def _blip_caption(image: Image.Image) -> dict:
    """Generate caption using BLIP model."""
    _ensure_model()

    if _model is None:
        # Fallback mock for environments without GPU/transformers
        return {
            "status": "success",
            "caption": "A photograph showing various elements in a scene.",
            "model": "Mock (install transformers for real inference)",
            "confidence": "N/A",
        }

    import torch
    inputs = _processor(image, return_tensors="pt").to(_device)
    with torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=100, num_beams=5)
    caption = _processor.decode(out[0], skip_special_tokens=True)
    return {
        "status": "success",
        "caption": caption,
        "model": "BLIP (Salesforce/blip-image-captioning-base)",
        "confidence": "High",
    }


def _refine_with_llm(caption: str) -> str:
    """
    Refine a raw BLIP caption into a human-like description using an LLM.
    Supports OpenAI and Groq. Falls back gracefully if no API key is set.
    """
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        return _local_refinement(caption)

    # Try OpenAI
    if os.getenv("OPENAI_API_KEY"):
        return _refine_openai(caption)

    # Try Groq
    if os.getenv("GROQ_API_KEY"):
        return _refine_groq(caption)

    return _local_refinement(caption)


def _refine_openai(caption: str) -> str:
    try:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert visual analyst. Given a raw image caption, "
                        "expand it into a rich, natural, human-like 2-3 sentence description "
                        "that conveys scene context, mood, and meaningful details."
                    ),
                },
                {"role": "user", "content": f"Raw caption: {caption}\n\nExpanded description:"},
            ],
            max_tokens=200,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"OpenAI refinement failed: {e}")
        return _local_refinement(caption)


def _refine_groq(caption: str) -> str:
    try:
        from groq import Groq
        client = Groq()
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert visual analyst. Given a raw image caption, "
                        "expand it into a rich, natural, human-like 2-3 sentence description "
                        "that conveys scene context, mood, and meaningful details."
                    ),
                },
                {"role": "user", "content": f"Raw caption: {caption}\n\nExpanded description:"},
            ],
            max_tokens=200,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Groq refinement failed: {e}")
        return _local_refinement(caption)


def _local_refinement(caption: str) -> str:
    """
    Simple rule-based local refinement when no LLM API is available.
    Adds contextual framing to the raw caption.
    """
    return (
        f"The image depicts {caption.lower().rstrip('.')}. "
        "This scene captures a visually rich moment with various elements that together form a coherent visual narrative. "
        "The composition and lighting contribute to the overall context and atmosphere of the image."
    )


def generate_caption(image: Image.Image, use_llm: bool = True) -> dict:
    """
    Main caption generation function.

    Args:
        image: Preprocessed PIL Image
        use_llm: Whether to refine caption with LLM

    Returns:
        dict with keys: status, caption, model, confidence, refined_caption (optional)
    """
    try:
        result = _blip_caption(image)
        if result["status"] != "success":
            return result

        if use_llm:
            refined = _refine_with_llm(result["caption"])
            result["refined_caption"] = refined

        return result

    except Exception as e:
        logger.error(f"Caption generation error: {e}")
        return {"status": "error", "error": str(e)}
