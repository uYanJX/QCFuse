"""Official BIRD execution-accuracy evaluation for Blend experiments."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


BIRD_SAMPLE_COUNT = 500

_MARKDOWN_FENCE_RE = re.compile(
    r"```(?:[A-Za-z0-9_.+-]+)?[ \t]*\r?\n(?P<body>.*?)\r?\n?```",
    re.DOTALL,
)
_MARKDOWN_FENCE_OPEN_RE = re.compile(
    r"```(?:[A-Za-z0-9_.+-]+)?[ \t]*\r?\n"
)


def _strip_markdown_fence(value: str) -> str:
    """Extract the final fenced SQL, including one missing its closing fence."""
    stripped = value.strip()
    matches = list(_MARKDOWN_FENCE_RE.finditer(stripped))
    if matches:
        return matches[-1].group("body").strip()
    openers = list(_MARKDOWN_FENCE_OPEN_RE.finditer(stripped))
    if openers:
        return stripped[openers[-1].end() :].strip()
    return stripped


def _answers(example: Mapping[str, Any]) -> list[str]:
    answers = example.get("answers", [])
    if isinstance(answers, str):
        answers = [answers]
    if not isinstance(answers, list):
        raise ValueError("BIRD example answers must be a list or string")
    return [str(answer) for answer in answers]


def _first_existing(candidates: Sequence[Path], description: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"{description} not found; checked: {checked}")


def _find_bird_database(database_root: Path, db_id: str) -> Path:
    candidates = (
        database_root / db_id / f"{db_id}.sqlite",
        database_root / db_id / f"{db_id}.db",
        database_root / db_id / "sqlite" / f"{db_id}.sqlite",
        database_root / f"{db_id}.sqlite",
        database_root / f"{db_id}.db",
    )
    matches = [path for path in candidates if path.is_file()]
    if not matches and (database_root / db_id).is_dir():
        matches = sorted(
            path
            for path in (database_root / db_id).rglob("*")
            if path.is_file() and path.suffix.casefold() in {".sqlite", ".db"}
        )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"BIRD database {db_id!r}: expected one SQLite file under "
            f"{database_root}, found {[str(path) for path in matches]}"
        )
    return matches[0]


class BirdExecutionEvaluator:
    """BIRD Mini-Dev Execution Accuracy using the official set comparison."""

    metric_name = "EX"

    def __init__(
        self,
        dataset: Sequence[Mapping[str, Any]],
        source_path: str | Path,
        database_root: str | Path,
        *,
        timeout_s: float = 30.0,
        expected_rows: int = BIRD_SAMPLE_COUNT,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("BIRD evaluation timeout must be positive")

        self.source_path = Path(source_path).resolve()
        self.database_root = Path(database_root).resolve()
        records = json.loads(self.source_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("BIRD source must contain a JSON array")
        if len(records) != expected_rows:
            raise ValueError(
                f"BIRD source must contain {expected_rows} rows, found {len(records)}"
            )
        if len(dataset) > len(records):
            raise ValueError("BIRD runtime dataset is longer than its official source")

        self.timeout_s = float(timeout_s)
        self._gold_sqls: list[str] = []
        self._database_paths: list[Path] = []
        for index, example in enumerate(dataset):
            source_index = int(example.get("_source_index", index))
            if source_index < 0 or source_index >= len(records):
                raise IndexError(f"BIRD source index out of range: {source_index}")
            record = records[source_index]
            question = str(record.get("question", ""))
            gold_sql = str(record.get("SQL", record.get("sql", ""))).strip()
            db_id = str(record.get("db_id", "")).strip()
            if str(example.get("input", "")) != question:
                raise ValueError(
                    f"BIRD row {index} (source {source_index}) question does not match source"
                )
            if _answers(example) != [gold_sql]:
                raise ValueError(
                    f"BIRD row {index} (source {source_index}) gold SQL does not match source"
                )
            if not db_id or not gold_sql:
                raise ValueError(
                    f"BIRD row {index} (source {source_index}) has incomplete source metadata"
                )
            self._gold_sqls.append(gold_sql)
            self._database_paths.append(
                _find_bird_database(self.database_root, db_id)
            )

    @property
    def description(self) -> str:
        return (
            f"BIRD EX | source={self.source_path} | "
            f"db_root={self.database_root}"
        )

    def score(
        self, index: int, prediction: str, ground_truths: Sequence[str]
    ) -> float:
        if index < 0 or index >= len(self._gold_sqls):
            raise IndexError(f"BIRD sample index out of range: {index}")
        if list(ground_truths) != [self._gold_sqls[index]]:
            raise ValueError(f"BIRD row {index} ground truth changed after alignment")

        predicted_sql = _strip_markdown_fence(prediction)
        if not predicted_sql:
            return 0.0

        deadline = time.monotonic() + self.timeout_s

        def stop_after_deadline() -> int:
            return int(time.monotonic() >= deadline)

        database_path = self._database_paths[index]
        uri = f"file:{database_path}?mode=ro&immutable=1"
        try:
            with sqlite3.connect(uri, uri=True) as connection:
                connection.execute("PRAGMA query_only = ON")
                connection.set_progress_handler(stop_after_deadline, 10_000)
                predicted_rows = connection.execute(predicted_sql).fetchall()
                gold_rows = connection.execute(self._gold_sqls[index]).fetchall()
        except (sqlite3.Error, sqlite3.Warning, TypeError, ValueError):
            return 0.0
        return float(set(predicted_rows) == set(gold_rows))


def create_bird_evaluator(
    dataset: Sequence[Mapping[str, Any]],
    data_dir: str | Path,
    *,
    bird_data: str | Path | None = None,
    bird_db_root: str | Path | None = None,
    timeout_s: float = 30.0,
) -> BirdExecutionEvaluator:
    """Create the evaluator, auto-discovering official files when possible."""

    resolved_data_dir = Path(data_dir).resolve()
    structured_candidates = (
        resolved_data_dir / "sources" / "structured",
        resolved_data_dir.parent / "sources" / "structured",
    )
    structured_root = next(
        (path for path in structured_candidates if path.is_dir()),
        structured_candidates[-1],
    )
    bird_root = structured_root / "bird_mini_dev"
    source_path = (
        Path(bird_data)
        if bird_data is not None
        else _first_existing(
            (
                bird_root / "minidev" / "MINIDEV" / "mini_dev_sqlite.json",
                bird_root / "mini_dev_data" / "mini_dev_sqlite.json",
                bird_root / "mini_dev_sqlite.json",
            ),
            "official BIRD Mini-Dev JSON",
        )
    )
    database_root = (
        Path(bird_db_root)
        if bird_db_root is not None
        else _first_existing(
            (
                bird_root / "minidev" / "MINIDEV" / "dev_databases",
                bird_root / "mini_dev_data" / "dev_databases",
                bird_root / "dev_databases",
            ),
            "BIRD SQLite database root",
        )
    )
    return BirdExecutionEvaluator(
        dataset,
        source_path,
        database_root,
        timeout_s=timeout_s,
    )


__all__ = ["BirdExecutionEvaluator", "create_bird_evaluator"]
