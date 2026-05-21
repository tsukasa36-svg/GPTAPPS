from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re


ENTITY_PATTERN = re.compile(r"#\d+\s*=\s*([A-Z0-9_]+)\s*\(", re.IGNORECASE)
SCHEMA_PATTERN = re.compile(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']+)'", re.IGNORECASE)


@dataclass(frozen=True)
class StepDocument:
    """Lightweight representation of a STEP file.

    This parser intentionally extracts metadata and entity counts only. Exact
    geometry should later be handled through OpenCascade, CadQuery, or another
    CAD kernel.
    """

    path: Path
    text: str
    schema: str | None
    entity_counts: Counter[str]

    @property
    def entity_count(self) -> int:
        return sum(self.entity_counts.values())

    @property
    def line_count(self) -> int:
        return self.text.count("\n") + 1

    @property
    def file_size_bytes(self) -> int:
        return self.path.stat().st_size


def read_step(path: str | Path) -> StepDocument:
    step_path = Path(path).expanduser().resolve()
    if not step_path.exists():
        raise FileNotFoundError(step_path)
    if step_path.suffix.lower() not in {".step", ".stp"}:
        raise ValueError(f"Expected a STEP/STP file, got: {step_path}")

    text = step_path.read_text(encoding="utf-8", errors="ignore")
    schema_match = SCHEMA_PATTERN.search(text)
    schema = schema_match.group(1).upper() if schema_match else None

    counts: Counter[str] = Counter()
    for match in ENTITY_PATTERN.finditer(text):
        counts[match.group(1).upper()] += 1

    return StepDocument(
        path=step_path,
        text=text,
        schema=schema,
        entity_counts=counts,
    )


def summarize_step(path: str | Path) -> dict[str, object]:
    doc = read_step(path)
    return {
        "path": str(doc.path),
        "schema": doc.schema,
        "file_size_bytes": doc.file_size_bytes,
        "line_count": doc.line_count,
        "entity_count": doc.entity_count,
        "top_entities": doc.entity_counts.most_common(20),
    }
