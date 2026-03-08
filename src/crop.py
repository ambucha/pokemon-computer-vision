import numpy as np
from PIL import Image


def crop_card(img: Image.Image, chroma_threshold: int = 15, padding: int = 4) -> Image.Image:
    """
    Detects the card by its color (blue card vs neutral gray background).
    Background has near-zero chroma; card has high chroma.
    Falls back to the original image if detection fails.
    """
    arr = np.array(img.convert('RGB')).astype(int)
    chroma = arr.max(axis=2) - arr.min(axis=2)   # 0 for gray, high for colored
    mask = chroma > chroma_threshold

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return img                               # fallback

    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]

    h, w = arr.shape[:2]
    y0 = max(0,   y0 - padding)
    y1 = min(h-1, y1 + padding)
    x0 = max(0,   x0 - padding)
    x1 = min(w-1, x1 + padding)

    return img.crop((x0, y0, x1, y1))
