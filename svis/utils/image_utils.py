"""
Image preprocessing and validation utilities.
"""

from PIL import Image, ImageOps
import numpy as np


def validate_image(image: Image.Image) -> tuple[bool, str]:
    """
    Validate image for suitability.

    Returns:
        (is_valid: bool, message: str)
    """
    w, h = image.size

    if w < 32 or h < 32:
        return False, f"Image too small ({w}x{h}). Minimum size is 32x32 pixels."

    if w > 4096 or h > 4096:
        return True, f"Large image ({w}x{h}) — will be resized for inference."

    return True, "OK"


def preprocess_image(
    image: Image.Image,
    target_size: tuple[int, int] = (384, 384),
    normalize: bool = False,
) -> Image.Image:
    """
    Preprocess image for model inference.

    Steps:
    1. Convert to RGB
    2. Apply EXIF orientation correction
    3. Resize to target size with aspect-ratio preservation
    4. Optionally normalize pixel values

    Args:
        image: Input PIL Image
        target_size: (width, height) to resize to
        normalize: Whether to return a numpy array normalized to [0, 1]

    Returns:
        Preprocessed PIL Image (or numpy array if normalize=True)
    """
    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Correct EXIF orientation
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    # Resize preserving aspect ratio, then pad/crop to target
    image = ImageOps.fit(image, target_size, method=Image.Resampling.LANCZOS)

    if normalize:
        arr = np.array(image).astype(np.float32) / 255.0
        return arr

    return image


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    """Convert PIL Image to bytes."""
    import io
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def get_image_stats(image: Image.Image) -> dict:
    """
    Compute basic statistics for an image.

    Returns dict with: width, height, mode, mean_brightness, is_grayscale
    """
    arr = np.array(image.convert("RGB"))
    brightness = float(arr.mean())
    r_std = float(arr[:, :, 0].std())
    g_std = float(arr[:, :, 1].std())
    b_std = float(arr[:, :, 2].std())
    is_grayscale = (r_std < 5 and g_std < 5 and b_std < 5)

    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "mean_brightness": round(brightness, 2),
        "is_grayscale": is_grayscale,
        "aspect_ratio": round(image.width / image.height, 2),
    }
