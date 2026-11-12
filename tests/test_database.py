from pathlib import Path

from dermavision.database import AnalysisRepository
from dermavision.model import Prediction
from dermavision.quality import QualityResult


def test_repository_round_trip(tmp_path: Path):
    repository = AnalysisRepository(tmp_path / "test.db")
    prediction = Prediction(
        predicted_code="nv",
        confidence=0.8,
        attention_score=0.1,
        requires_attention=False,
        is_confident=True,
        uncertainty=0.2,
        probabilities={
            "akiec": 0.02,
            "bcc": 0.03,
            "bkl": 0.05,
            "df": 0.03,
            "mel": 0.05,
            "nv": 0.80,
            "vasc": 0.02,
        },
        device="cpu",
    )
    quality = QualityResult(90, 100.0, 125.0, 30.0, 600, 450, True, (), False)

    record_id = repository.save(
        image_hash="abc",
        filename="test.jpg",
        prediction=prediction,
        quality=quality,
    )

    assert record_id == 1
    assert repository.count() == 1
    assert repository.recent()[0]["predicted_code"] == "nv"
