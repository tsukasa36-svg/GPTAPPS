from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from cad_ai_suite.cad.feature_extraction import extract_step_features
from cad_ai_suite.cad.step_io import summarize_step


DEFAULT_INBOX = Path("step_inbox")
DEFAULT_RAW_DIR = Path("data/raw_step")
DEFAULT_MANIFEST = Path("data/manifests/step_inventory.json")


@dataclass(frozen=True)
class ImportedStep:
    source_path: str
    raw_path: str
    schema: str | None
    entity_count: int
    file_size_bytes: int
    status: str
    imported_at: str


def import_step_folder(
    inbox: str | Path = DEFAULT_INBOX,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    move_after_import: bool = False,
) -> list[ImportedStep]:
    inbox_path = Path(inbox)
    raw_path = Path(raw_dir)
    manifest = Path(manifest_path)
    raw_path.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    step_files = sorted(
        path
        for path in inbox_path.iterdir()
        if path.is_file() and path.suffix.lower() in {".step", ".stp"}
    )

    records: list[ImportedStep] = []
    for source in step_files:
        destination = _unique_destination(raw_path / source.name)
        try:
            shutil.copy2(source, destination)
            summary = summarize_step(destination)
            extract_step_features(destination)
            status = "imported"
            if move_after_import:
                imported_dir = inbox_path / "imported"
                imported_dir.mkdir(exist_ok=True)
                shutil.move(str(source), imported_dir / source.name)
            records.append(
                ImportedStep(
                    source_path=str(source.resolve()),
                    raw_path=str(destination.resolve()),
                    schema=summary["schema"],
                    entity_count=int(summary["entity_count"]),
                    file_size_bytes=int(summary["file_size_bytes"]),
                    status=status,
                    imported_at=datetime.now(timezone.utc).isoformat(),
                )
            )
        except Exception:
            rejected_dir = inbox_path / "rejected"
            rejected_dir.mkdir(exist_ok=True)
            if move_after_import and source.exists():
                shutil.move(str(source), rejected_dir / source.name)
            records.append(
                ImportedStep(
                    source_path=str(source.resolve()),
                    raw_path="",
                    schema=None,
                    entity_count=0,
                    file_size_bytes=source.stat().st_size if source.exists() else 0,
                    status="rejected",
                    imported_at=datetime.now(timezone.utc).isoformat(),
                )
            )

    existing = _load_manifest(manifest)
    existing.extend(asdict(record) for record in records)
    manifest.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return records


def _load_manifest(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    index = 2
    while True:
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Import dropped STEP/STP files into the CAD AI workspace.")
    parser.add_argument("--inbox", default=str(DEFAULT_INBOX))
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move source files into step_inbox/imported after copying them into data/raw_step.",
    )
    args = parser.parse_args()

    records = import_step_folder(
        inbox=args.inbox,
        raw_dir=args.raw_dir,
        manifest_path=args.manifest,
        move_after_import=args.move,
    )
    if not records:
        print(f"No STEP/STP files found in {args.inbox}")
        return
    for record in records:
        print(f"{record.status}: {record.source_path} -> {record.raw_path}")


if __name__ == "__main__":
    main()
