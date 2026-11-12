from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class QualityResult:
    score: int
    sharpness: float
    brightness: float
    contrast: float
    width: int
    height: int
    looks_like_skin: bool
    issues: tuple[str, ...]
    blocking: bool

    def to_dict(self) -> dict:
        result = asdict(self)
        result["issues"] = list(self.issues)
        return result


def _roughness(rgb_float: np.ndarray) -> float:
    grey = rgb_float.mean(axis=2)
    across = np.abs(grey[:, 1:] - grey[:, :-1]).mean()
    down = np.abs(grey[1:, :] - grey[:-1, :]).mean()
    return float((across + down) / 2)


def skin_photo_heuristic(image: Image.Image) -> bool:
    """Kontroll konservativ ndaj zhurmës, diagrameve dhe imazheve gri.

    Nuk është detektor i lëkurës dhe prandaj rezultati shfaqet vetëm si kontroll
    teknik. Kufijtë vijnë nga validimi i projektit burim të modelit.
    """
    resized = (
        np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
    )
    roughness = _roughness(resized)
    saturation = float((resized.max(axis=2) - resized.min(axis=2)).mean())
    red, green, blue = resized.mean(axis=(0, 1))
    red_dominant = red >= green - 0.05 and red >= blue - 0.05
    return bool(
        0.002 <= roughness <= 0.08
        and saturation >= 0.03
        and red_dominant
        and red >= 0.35
    )


def assess_image_quality(image: Image.Image) -> QualityResult:
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    brightness = float(grey.mean())
    contrast = float(grey.std())
    looks_like_skin = skin_photo_heuristic(image)

    score = 100
    issues: list[str] = []
    blocking = False

    if min(width, height) < 128:
        score -= 45
        issues.append("Rezolucioni është shumë i ulët (minimumi praktik: 128 px).")
        blocking = True
    elif min(width, height) < 224:
        score -= 18
        issues.append("Rezolucioni është nën 224 px; detajet mund të humbin.")

    if sharpness < 8:
        score -= 40
        issues.append("Fotografia është tepër e paqartë.")
        blocking = True
    elif sharpness < 35:
        score -= 22
        issues.append("Fotografia duket e paqartë; fokusohu më afër lezionit.")

    if brightness < 35:
        score -= 25
        issues.append("Fotografia është shumë e errët.")
    elif brightness > 225:
        score -= 25
        issues.append("Fotografia është e mbiekspozuar.")

    if contrast < 12:
        score -= 18
        issues.append("Kontrasti është shumë i ulët.")

    if not looks_like_skin:
        score -= 30
        issues.append(
            "Imazhi nuk kaloi kontrollin heuristik për foto dermatoskopike/lëkure."
        )

    return QualityResult(
        score=max(0, min(100, score)),
        sharpness=round(sharpness, 2),
        brightness=round(brightness, 2),
        contrast=round(contrast, 2),
        width=width,
        height=height,
        looks_like_skin=looks_like_skin,
        issues=tuple(issues),
        blocking=blocking,
    )
