from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import re


@dataclass(frozen=True)
class HolePattern:
    count: int = 0
    diameter_mm: float = 6.0


@dataclass(frozen=True)
class DesignSpec:
    kind: str
    material: str = "aluminum"
    length_mm: float = 80.0
    width_mm: float = 40.0
    height_mm: float = 80.0
    thickness_mm: float = 6.0
    hole_pattern: HolePattern = field(default_factory=HolePattern)
    gusset: bool = False
    source_prompt: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


MATERIALS = [
    "aluminum",
    "steel",
    "stainless steel",
    "plastic",
    "abs",
    "nylon",
    "titanium",
]


def parse_design_prompt(prompt: str) -> DesignSpec:
    text = prompt.lower()
    numbers = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*mm", text)]

    kind = "fixture_plate"
    if "l bracket" in text or "l-bracket" in text or "angle bracket" in text:
        kind = "l_bracket"
    elif "fixture" in text:
        kind = "fixture_plate"
    elif "plate" in text:
        kind = "mounting_plate"

    material = next((candidate for candidate in MATERIALS if candidate in text), "aluminum")

    length = numbers[0] if len(numbers) >= 1 else 80.0
    height = numbers[1] if len(numbers) >= 2 else 80.0
    thickness = _extract_named_dimension(text, ["thick", "thickness"], default=numbers[2] if len(numbers) >= 3 else 6.0)
    width = _extract_named_dimension(text, ["wide", "width"], default=40.0)

    hole_count = _extract_hole_count(text)
    hole_diameter = _extract_metric_thread_diameter(text) or _extract_named_dimension(
        text,
        ["hole", "holes"],
        default=6.0,
    )

    return DesignSpec(
        kind=kind,
        material=material,
        length_mm=length,
        width_mm=width,
        height_mm=height,
        thickness_mm=thickness,
        hole_pattern=HolePattern(count=hole_count, diameter_mm=hole_diameter),
        gusset=("gusset" in text or "triangular" in text),
        source_prompt=prompt,
    )


def export_design(spec: DesignSpec, output_dir: str | Path = "data/generated") -> dict[str, Path | str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = _safe_stem(spec.kind)
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")

    result: dict[str, Path | str] = {"json": json_path}
    try:
        step_path = out_dir / f"{stem}.step"
        _export_with_cadquery(spec, step_path)
        result["step"] = step_path
    except Exception as exc:  # CadQuery is optional in this prototype.
        result["step_error"] = str(exc)
    return result


def _extract_named_dimension(text: str, names: list[str], default: float) -> float:
    for name in names:
        match = re.search(rf"(\d+(?:\.\d+)?)\s*mm\s*{re.escape(name)}", text)
        if match:
            return float(match.group(1))
        match = re.search(rf"{re.escape(name)}\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)\s*mm", text)
        if match:
            return float(match.group(1))
    return float(default)


def _extract_hole_count(text: str) -> int:
    word_counts = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "eight": 8,
    }
    match = re.search(r"(?<![a-zA-Z])(\d+)\s*(?:x\s*)?(?:m\d+\s*)?(?:mounting\s*)?holes?", text)
    if match:
        return int(match.group(1))
    for word, count in word_counts.items():
        if re.search(rf"\b{word}\b.*\bholes?\b", text):
            return count
    return 0


def _extract_metric_thread_diameter(text: str) -> float | None:
    match = re.search(r"\bm(\d+(?:\.\d+)?)\b", text)
    return float(match.group(1)) if match else None


def _safe_stem(kind: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", kind.lower()).strip("_") or "generated_part"


def _export_with_cadquery(spec: DesignSpec, step_path: Path) -> None:
    try:
        import cadquery as cq
    except ImportError as exc:
        raise RuntimeError(
            "CadQuery is not installed. Install it with: pip install -e \".[cad]\""
        ) from exc

    if spec.kind == "l_bracket":
        solid = _build_l_bracket(cq, spec)
    else:
        solid = _build_plate(cq, spec)

    cq.exporters.export(solid, str(step_path))


def _build_l_bracket(cq, spec: DesignSpec):
    length = spec.length_mm
    width = spec.width_mm
    height = spec.height_mm
    thickness = spec.thickness_mm
    hole_diameter = spec.hole_pattern.diameter_mm

    base = cq.Workplane("XY").box(length, width, thickness).translate((length / 2, 0, thickness / 2))
    upright = cq.Workplane("XY").box(thickness, width, height).translate((thickness / 2, 0, height / 2))
    solid = base.union(upright)

    points = _hole_points(spec.hole_pattern.count, length, width)
    if points:
        solid = solid.faces(">Z").workplane().pushPoints(points).hole(hole_diameter)
    return solid


def _build_plate(cq, spec: DesignSpec):
    length = spec.length_mm
    width = spec.width_mm
    thickness = spec.thickness_mm
    hole_diameter = spec.hole_pattern.diameter_mm

    solid = cq.Workplane("XY").box(length, width, thickness).translate((length / 2, 0, thickness / 2))
    points = _hole_points(spec.hole_pattern.count, length, width)
    if points:
        solid = solid.faces(">Z").workplane().pushPoints(points).hole(hole_diameter)
    return solid


def _hole_points(count: int, length: float, width: float) -> list[tuple[float, float]]:
    if count <= 0:
        return []
    x_positions = [length * 0.25, length * 0.75]
    y_positions = [-width * 0.25, width * 0.25]
    points = [(x, y) for x in x_positions for y in y_positions]
    if count == 1:
        return [(length * 0.5, 0.0)]
    if count == 2:
        return [(length * 0.3, 0.0), (length * 0.7, 0.0)]
    return points[: min(count, len(points))]
