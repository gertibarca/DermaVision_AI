from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18

from .config import (
    ATTENTION_CLASSES,
    ATTENTION_THRESHOLD,
    CLASSES,
    DIAGNOSIS_CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_PATH,
    TTA_QUARTER_TURNS,
)


@dataclass(frozen=True)
class Prediction:
    predicted_code: str
    confidence: float
    attention_score: float
    requires_attention: bool
    is_confident: bool
    uncertainty: float
    probabilities: dict[str, float]
    device: str

    @property
    def uncertainty_label(self) -> str:
        if self.uncertainty < 0.35 and self.confidence >= 0.70:
            return "E ulët"
        if self.uncertainty < 0.65:
            return "Mesatare"
        return "E lartë"


class ModelService:
    """Ngarkon modelin një herë dhe ofron inferencë + Grad-CAM thread-safe."""

    def __init__(self, model_path=MODEL_PATH, prefer_cuda: bool = True):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Peshat e modelit mungojnë: {model_path}. Ekzekuto download_model.py."
            )

        self.device = torch.device(
            "cuda" if prefer_cuda and torch.cuda.is_available() else "cpu"
        )
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(CLASSES))

        # weights_only=True shmang ekzekutimin e objekteve arbitrare nga skedari .pth.
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        self.to_tensor = transforms.Compose(
            [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
        )
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self._lock = threading.RLock()

    def _normalized(self, image: Image.Image) -> torch.Tensor:
        return self.normalize(self.to_tensor(image.convert("RGB")))

    def predict(self, image: Image.Image) -> Prediction:
        base = self._normalized(image)
        batch = torch.stack(
            [torch.rot90(base, k=k, dims=(1, 2)) for k in TTA_QUARTER_TURNS]
        ).to(self.device)

        with self._lock, torch.inference_mode():
            probs_tensor = torch.softmax(self.model(batch), dim=1).mean(dim=0).cpu()

        probs = {code: float(value) for code, value in zip(CLASSES, probs_tensor)}
        top_index = int(probs_tensor.argmax())
        confidence = float(probs_tensor[top_index])
        attention_score = float(sum(probs[code] for code in ATTENTION_CLASSES))

        values = np.clip(np.asarray(list(probs.values()), dtype=float), 1e-12, 1.0)
        entropy = float(-(values * np.log(values)).sum() / math.log(len(CLASSES)))

        return Prediction(
            predicted_code=CLASSES[top_index],
            confidence=confidence,
            attention_score=attention_score,
            requires_attention=attention_score >= ATTENTION_THRESHOLD,
            is_confident=confidence >= DIAGNOSIS_CONFIDENCE_THRESHOLD,
            uncertainty=entropy,
            probabilities=probs,
            device=str(self.device),
        )

    def gradcam(
        self, image: Image.Image, class_code: str
    ) -> tuple[Image.Image, Image.Image]:
        """Kthen heatmap dhe mbivendosje Grad-CAM për klasën e kërkuar."""
        class_index = CLASSES.index(class_code)
        prepared = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        batch = self._normalized(prepared).unsqueeze(0).to(self.device)
        captured: dict[str, torch.Tensor] = {}

        def save_activations(_module, _inputs, output):
            captured["activations"] = output

        with self._lock:
            handle = self.model.layer4.register_forward_hook(save_activations)
            try:
                logits = self.model(batch)
                activations = captured["activations"]
                gradients = torch.autograd.grad(logits[0, class_index], activations)[0]
            finally:
                handle.remove()

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations.detach()).sum(dim=1).squeeze(0))
        if float(cam.max()) > 0:
            cam = cam / cam.max()

        cam_np = cam.cpu().numpy()
        heat = Image.fromarray(np.uint8(cam_np * 255), mode="L").resize(
            prepared.size, Image.Resampling.BICUBIC
        )

        # Paletë Jet e lehtë: blu = ndikim i ulët, e kuqe = ndikim i lartë.
        values = np.asarray(heat, dtype=float) / 255.0
        stops = np.array([0.0, 0.125, 0.375, 0.625, 0.875, 1.0])
        colors = np.array(
            [
                [0, 0, 143],
                [0, 0, 255],
                [0, 255, 255],
                [255, 255, 0],
                [255, 0, 0],
                [128, 0, 0],
            ]
        )
        channels = [
            np.interp(values, stops, colors[:, channel]) for channel in range(3)
        ]
        colorized = Image.fromarray(
            np.stack(channels, axis=-1).astype(np.uint8), mode="RGB"
        )
        overlay = Image.blend(prepared, colorized, alpha=0.48)
        return colorized, overlay

    @property
    def device_label(self) -> str:
        if self.device.type == "cuda":
            return f"GPU · {torch.cuda.get_device_name(0)}"
        return "CPU"
