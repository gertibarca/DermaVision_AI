import numpy as np
from PIL import Image

from dermavision.quality import assess_image_quality


def test_tiny_image_is_blocked():
    image = Image.new("RGB", (64, 64), color=(180, 120, 110))
    result = assess_image_quality(image)
    assert result.blocking is True
    assert result.score < 60


def test_quality_score_stays_in_range():
    rng = np.random.default_rng(42)
    pixels = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
    result = assess_image_quality(Image.fromarray(pixels))
    assert 0 <= result.score <= 100
