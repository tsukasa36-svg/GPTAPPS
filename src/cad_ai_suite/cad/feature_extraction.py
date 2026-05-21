from __future__ import annotations

from pathlib import Path
from typing import Iterable

from cad_ai_suite.cad.step_io import read_step


FEATURE_KEYS = [
    "file_size_bytes",
    "line_count",
    "entity_count",
    "cartesian_point_count",
    "direction_count",
    "vector_count",
    "axis2_placement_3d_count",
    "plane_count",
    "cylindrical_surface_count",
    "conical_surface_count",
    "spherical_surface_count",
    "toroidal_surface_count",
    "circle_count",
    "ellipse_count",
    "edge_curve_count",
    "oriented_edge_count",
    "advanced_face_count",
    "face_bound_count",
    "closed_shell_count",
    "manifold_solid_brep_count",
    "faceted_brep_count",
    "boolean_result_count",
    "product_count",
    "shape_definition_count",
    "hole_signal",
    "curved_surface_signal",
    "solid_signal",
]


ENTITY_TO_FEATURE = {
    "CARTESIAN_POINT": "cartesian_point_count",
    "DIRECTION": "direction_count",
    "VECTOR": "vector_count",
    "AXIS2_PLACEMENT_3D": "axis2_placement_3d_count",
    "PLANE": "plane_count",
    "CYLINDRICAL_SURFACE": "cylindrical_surface_count",
    "CONICAL_SURFACE": "conical_surface_count",
    "SPHERICAL_SURFACE": "spherical_surface_count",
    "TOROIDAL_SURFACE": "toroidal_surface_count",
    "CIRCLE": "circle_count",
    "ELLIPSE": "ellipse_count",
    "EDGE_CURVE": "edge_curve_count",
    "ORIENTED_EDGE": "oriented_edge_count",
    "ADVANCED_FACE": "advanced_face_count",
    "FACE_BOUND": "face_bound_count",
    "CLOSED_SHELL": "closed_shell_count",
    "MANIFOLD_SOLID_BREP": "manifold_solid_brep_count",
    "FACETED_BREP": "faceted_brep_count",
    "BOOLEAN_RESULT": "boolean_result_count",
    "PRODUCT": "product_count",
    "SHAPE_DEFINITION_REPRESENTATION": "shape_definition_count",
}


def extract_step_features(path: str | Path) -> dict[str, float]:
    doc = read_step(path)
    features = {key: 0.0 for key in FEATURE_KEYS}
    features["file_size_bytes"] = float(doc.file_size_bytes)
    features["line_count"] = float(doc.line_count)
    features["entity_count"] = float(doc.entity_count)

    for entity_name, feature_name in ENTITY_TO_FEATURE.items():
        features[feature_name] = float(doc.entity_counts.get(entity_name, 0))

    features["hole_signal"] = (
        features["circle_count"] + features["cylindrical_surface_count"]
    )
    features["curved_surface_signal"] = (
        features["cylindrical_surface_count"]
        + features["conical_surface_count"]
        + features["spherical_surface_count"]
        + features["toroidal_surface_count"]
    )
    features["solid_signal"] = (
        features["manifold_solid_brep_count"]
        + features["faceted_brep_count"]
        + features["closed_shell_count"]
    )
    return features


def vectorize_features(features: dict[str, float]) -> list[float]:
    return [float(features.get(key, 0.0)) for key in FEATURE_KEYS]


def normalize_rows(rows: Iterable[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    matrix = [list(row) for row in rows]
    if not matrix:
        return [], [], []

    columns = list(zip(*matrix))
    means = [sum(col) / len(col) for col in columns]
    stds = []
    for col, mean in zip(columns, means):
        variance = sum((value - mean) ** 2 for value in col) / max(len(col), 1)
        stds.append(variance**0.5 or 1.0)

    normalized = [
        [(value - mean) / std for value, mean, std in zip(row, means, stds)]
        for row in matrix
    ]
    return normalized, means, stds
