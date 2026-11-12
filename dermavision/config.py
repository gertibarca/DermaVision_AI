from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(
    os.getenv("DERMAVISION_MODEL_PATH", BASE_DIR / "models" / "resnet18_ham10000.pth")
)
DATABASE_PATH = Path(
    os.getenv("DERMAVISION_DATABASE_PATH", BASE_DIR / "data" / "dermavision.db")
)
SAMPLES_DIR = BASE_DIR / "samples"
MODEL_URL = (
    "https://huggingface.co/miloszkrawczyk/skin-lesion-classifier/"
    "resolve/main/resnet.pth?download=true"
)

APP_NAME = "DermaVision AI"
APP_VERSION = "1.0.0"
IMAGE_SIZE = 224
MAX_UPLOAD_MB = 8
MAX_IMAGE_PIXELS = 50_000_000

# Renditja eshte pjese e peshave te modelit dhe nuk duhet ndryshuar.
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
ATTENTION_CLASSES = ("akiec", "bcc", "mel")
ATTENTION_THRESHOLD = 0.15
DIAGNOSIS_CONFIDENCE_THRESHOLD = 0.60

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TTA_QUARTER_TURNS = (0, 1, 2, 3)


@dataclass(frozen=True)
class LesionInfo:
    name_sq: str
    name_en: str
    category: str
    description: str
    guidance: str
    color: str


LESION_INFO: dict[str, LesionInfo] = {
    "akiec": LesionInfo(
        "Keratozë aktinike / karcinomë intraepiteliale",
        "Actinic keratosis / intraepithelial carcinoma",
        "Kërkon vëmendje",
        "Ndryshim i lidhur shpesh me dëmtimin kronik nga dielli dhe me potencial parakanceroz.",
        "Rekomandohet vlerësim profesional nga dermatologu.",
        "#f59e0b",
    ),
    "bcc": LesionInfo(
        "Karcinomë bazocelulare",
        "Basal cell carcinoma",
        "Kërkon vëmendje",
        "Formë e zakonshme e kancerit të lëkurës që zakonisht rritet ngadalë.",
        "Kërkon kontroll te dermatologu; vetëm biopsia mund ta konfirmojë.",
        "#ef4444",
    ),
    "bkl": LesionInfo(
        "Lezion beninj i ngjashëm me keratozën",
        "Benign keratosis-like lesion",
        "Zakonisht beninj",
        "Grup ndryshimesh zakonisht jokanceroze, përfshirë keratozën seborreike.",
        "Kontrollo çdo ndryshim të ri, rritje, gjakderdhje ose kruarje.",
        "#22c55e",
    ),
    "df": LesionInfo(
        "Dermatofibromë",
        "Dermatofibroma",
        "Zakonisht beninj",
        "Nyjë e vogël dhe zakonisht beninje e lëkurës.",
        "Vlerësimi profesional rekomandohet nëse ndryshon ose shkakton shqetësim.",
        "#14b8a6",
    ),
    "mel": LesionInfo(
        "Melanomë",
        "Melanoma",
        "Kërkon vëmendje",
        "Formë serioze e kancerit të lëkurës që lidhet me qelizat prodhuese të pigmentit.",
        "Kërko vlerësim dermatologjik sa më shpejt; aplikacioni nuk jep diagnozë.",
        "#dc2626",
    ),
    "nv": LesionInfo(
        "Nevus melanocitar",
        "Melanocytic nevus",
        "Zakonisht beninj",
        "Nishan i zakonshëm i formuar nga qeliza melanocitare.",
        "Një nishan i ri ose që ndryshon duhet kontrolluar nga dermatologu.",
        "#3b82f6",
    ),
    "vasc": LesionInfo(
        "Lezion vaskular",
        "Vascular lesion",
        "Zakonisht beninj",
        "Ndryshim i lidhur me enët e gjakut, shpesh me ngjyrë të kuqe ose vjollcë.",
        "Kontroll profesional nëse rritet, gjakoset ose ndryshon.",
        "#8b5cf6",
    ),
}


BENCHMARK = {
    "test_images": 1508,
    "accuracy": 0.749,
    "attention_recall": 0.920,
    "attention_precision": 0.430,
    "missed_attention_cases": 25,
    "false_alarms": 383,
    "split": "Sipas lesion_id (pa rrjedhje të së njëjtës lezion mes train/test)",
}

CLASS_METRICS = {
    "akiec": {"support": 60, "recall": 0.533, "precision": 0.464},
    "bcc": {"support": 79, "recall": 0.722, "precision": 0.633},
    "bkl": {"support": 195, "recall": 0.456, "precision": 0.742},
    "df": {"support": 12, "recall": 0.750, "precision": 0.243},
    "mel": {"support": 175, "recall": 0.663, "precision": 0.425},
    "nv": {"support": 967, "recall": 0.834, "precision": 0.911},
    "vasc": {"support": 20, "recall": 1.000, "precision": 0.588},
}

SAMPLE_LABELS = {
    "ISIC_0025303.jpg": "mel",
    "ISIC_0025937.jpg": "bkl",
    "ISIC_0026092.jpg": "vasc",
    "ISIC_0027347.jpg": "nv",
    "ISIC_0030555.jpg": "df",
    "ISIC_0032384.jpg": "bcc",
}
