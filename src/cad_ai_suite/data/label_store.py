from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sqlite3
from typing import Iterable


DEFAULT_DB_PATH = Path("data/labels.db")


@dataclass(frozen=True)
class LabelRecord:
    id: int | None
    path: str
    label: str
    description: str
    tags: list[str]


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    init_db(connection)
    return connection


def init_db(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            label TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            tags_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.commit()


def upsert_label(
    path: str | Path,
    label: str,
    description: str = "",
    tags: Iterable[str] | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> LabelRecord:
    resolved_path = str(Path(path).expanduser().resolve())
    clean_tags = [tag.strip() for tag in (tags or []) if tag.strip()]

    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO labels (path, label, description, tags_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                label = excluded.label,
                description = excluded.description,
                tags_json = excluded.tags_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (resolved_path, label.strip(), description.strip(), json.dumps(clean_tags)),
        )
        connection.commit()
        row = connection.execute(
            "SELECT * FROM labels WHERE path = ?",
            (resolved_path,),
        ).fetchone()

    return _row_to_record(row)


def list_labels(db_path: str | Path = DEFAULT_DB_PATH) -> list[LabelRecord]:
    with connect(db_path) as connection:
        rows = connection.execute("SELECT * FROM labels ORDER BY updated_at DESC").fetchall()
    return [_row_to_record(row) for row in rows]


def delete_label(path: str | Path, db_path: str | Path = DEFAULT_DB_PATH) -> None:
    resolved_path = str(Path(path).expanduser().resolve())
    with connect(db_path) as connection:
        connection.execute("DELETE FROM labels WHERE path = ?", (resolved_path,))
        connection.commit()


def _row_to_record(row: sqlite3.Row) -> LabelRecord:
    return LabelRecord(
        id=int(row["id"]),
        path=str(row["path"]),
        label=str(row["label"]),
        description=str(row["description"]),
        tags=list(json.loads(row["tags_json"])),
    )
