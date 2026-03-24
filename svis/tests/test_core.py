"""
Unit tests for SVIS core modules.
Run with: pytest tests/ -v
"""

import pytest
from PIL import Image
import numpy as np


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def sample_image():
    """Create a simple 224x224 RGB test image."""
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def tiny_image():
    arr = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def large_image():
    arr = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def gray_image():
    arr = np.full((224, 224, 3), 128, dtype=np.uint8)
    return Image.fromarray(arr)


# ── Image Utils Tests ─────────────────────────────────────────

class TestImageUtils:
    def test_validate_normal_image(self, sample_image):
        from utils.image_utils import validate_image
        valid, msg = validate_image(sample_image)
        assert valid is True

    def test_validate_tiny_image(self, tiny_image):
        from utils.image_utils import validate_image
        valid, msg = validate_image(tiny_image)
        assert valid is False
        assert "too small" in msg.lower()

    def test_validate_large_image(self, large_image):
        from utils.image_utils import validate_image
        valid, msg = validate_image(large_image)
        assert valid is True

    def test_preprocess_returns_pil(self, sample_image):
        from utils.image_utils import preprocess_image
        result = preprocess_image(sample_image)
        assert isinstance(result, Image.Image)

    def test_preprocess_target_size(self, sample_image):
        from utils.image_utils import preprocess_image
        result = preprocess_image(sample_image, target_size=(128, 128))
        assert result.size == (128, 128)

    def test_preprocess_rgb_mode(self, gray_image):
        from utils.image_utils import preprocess_image
        result = preprocess_image(gray_image)
        assert result.mode == "RGB"

    def test_preprocess_normalize(self, sample_image):
        from utils.image_utils import preprocess_image
        arr = preprocess_image(sample_image, normalize=True)
        assert isinstance(arr, np.ndarray)
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_get_image_stats(self, sample_image):
        from utils.image_utils import get_image_stats
        stats = get_image_stats(sample_image)
        assert "width" in stats
        assert "height" in stats
        assert "mean_brightness" in stats
        assert stats["width"] == 224
        assert stats["height"] == 224

    def test_get_image_stats_grayscale_detection(self, gray_image):
        from utils.image_utils import get_image_stats
        stats = get_image_stats(gray_image)
        assert stats["is_grayscale"] is True

    def test_image_to_bytes(self, sample_image):
        from utils.image_utils import image_to_bytes
        data = image_to_bytes(sample_image)
        assert isinstance(data, bytes)
        assert len(data) > 0


# ── Captioner Tests ───────────────────────────────────────────

class TestCaptioner:
    def test_generate_caption_returns_dict(self, sample_image):
        from models.captioner import generate_caption
        result = generate_caption(sample_image, use_llm=False)
        assert isinstance(result, dict)
        assert "status" in result

    def test_generate_caption_has_caption_key(self, sample_image):
        from models.captioner import generate_caption
        result = generate_caption(sample_image, use_llm=False)
        if result["status"] == "success":
            assert "caption" in result
            assert isinstance(result["caption"], str)
            assert len(result["caption"]) > 0

    def test_local_refinement(self):
        from models.captioner import _local_refinement
        raw = "a dog sitting on grass"
        refined = _local_refinement(raw)
        assert isinstance(refined, str)
        assert len(refined) > len(raw)

    def test_generate_caption_llm_mode(self, sample_image):
        from models.captioner import generate_caption
        result = generate_caption(sample_image, use_llm=True)
        assert isinstance(result, dict)
        assert "status" in result


# ── VQA Tests ─────────────────────────────────────────────────

class TestVQA:
    def test_answer_question_returns_string(self, sample_image):
        from models.vqa import answer_question
        answer = answer_question(sample_image, "A test image.", "What is in this image?")
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_mock_vqa_color_question(self):
        from models.vqa import _mock_vqa
        answer = _mock_vqa("What color is the background?")
        assert isinstance(answer, str)

    def test_mock_vqa_people_question(self):
        from models.vqa import _mock_vqa
        answer = _mock_vqa("Are there any people in this image?")
        assert "people" in answer.lower() or "person" in answer.lower() or "appear" in answer.lower()

    def test_local_expand(self):
        from models.vqa import _local_expand
        result = _local_expand("What color?", "red", "A red apple on a table.")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_vqa_with_empty_question(self, sample_image):
        from models.vqa import answer_question
        answer = answer_question(sample_image, "Some image.", "")
        assert isinstance(answer, str)


# ── Session Tests ─────────────────────────────────────────────

class TestSession:
    def test_init_session_state(self, monkeypatch):
        # Mock streamlit session state
        mock_state = {}

        class MockSS(dict):
            pass

        import streamlit as st
        monkeypatch.setattr(st, "session_state", MockSS())

        from utils.session import init_session_state
        init_session_state()

        assert "chat_history" in st.session_state
        assert "current_caption" in st.session_state
        assert isinstance(st.session_state["chat_history"], list)
