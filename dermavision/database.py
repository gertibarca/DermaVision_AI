from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .config import DATABASE_PATH
from .model import Prediction
from .quality import QualityResult


class AnalysisRepository:
    def __init__(self, path: Path = DATABASE_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    image_hash TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    predicted_code TEXT NOT NULL,
                    confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
                    attention_score REAL NOT NULL CHECK(attention_score BETWEEN 0 AND 1),
                    requires_attention INTEGER NOT NULL,
                    uncertainty REAL NOT NULL,
                    quality_score INTEGER NOT NULL CHECK(quality_score BETWEEN 0 AND 100),
                    probabilities_json TEXT NOT NULL,
                    quality_json TEXT NOT NULL,
                    age_group TEXT,
                    body_site TEXT,
                    sex TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at DESC)"
            )

    def save(
        self,
        *,
        image_hash: str,
        filename: str,
        prediction: Prediction,
        quality: QualityResult,
        metadata: dict[str, str] | None = None,
    ) -> int:
        metadata = metadata or {}
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO analyses (
                    created_at, image_hash, filename, predicted_code, confidence,
                    attention_score, requires_attention, uncertainty, quality_score,
                    probabilities_json, quality_json, age_group, body_site, sex
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(timespec="seconds"),
                    image_hash,
                    filename,
                    prediction.predicted_code,
                    prediction.confidence,
                    prediction.attention_score,
                    int(prediction.requires_attention),
                    prediction.uncertainty,
                    quality.score,
                    json.dumps(prediction.probabilities, ensure_ascii=False),
                    json.dumps(quality.to_dict(), ensure_ascii=False),
                    metadata.get("age_group", ""),
                    metadata.get("body_site", ""),
                    metadata.get("sex", ""),
                ),
            )
            return int(cursor.lastrowid)

    def recent(self, limit: int = 250) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT id, created_at, filename, predicted_code, confidence,
                       attention_score, requires_attention, uncertainty,
                       quality_score, age_group, body_site, sex
                FROM analyses ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def count(self) -> int:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM analyses"
            ).fetchone()
        return int(row["total"])

    def clear(self) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM analyses")
