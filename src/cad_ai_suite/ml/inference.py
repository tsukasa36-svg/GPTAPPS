from __future__ import annotations

from pathlib import Path

from cad_ai_suite.cad.feature_extraction import FEATURE_KEYS, extract_step_features, vectorize_features
from cad_ai_suite.ml.dataset import normalize_single_feature_row
from cad_ai_suite.ml.model import build_classifier, require_torch


def classify_step(path: str | Path, model_path: str | Path) -> list[tuple[str, float]]:
    torch, _ = require_torch()
    checkpoint = torch.load(model_path, map_location="cpu")
    label_names = list(checkpoint["label_names"])
    model = build_classifier(
        input_size=len(FEATURE_KEYS),
        output_size=len(label_names),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    raw = vectorize_features(extract_step_features(path))
    normalized = normalize_single_feature_row(
        raw,
        list(checkpoint["means"]),
        list(checkpoint["stds"]),
    )
    x = torch.tensor([normalized], dtype=torch.float32)
    with torch.no_grad():
        probabilities = torch.softmax(model(x), dim=1)[0].tolist()

    ranked = sorted(zip(label_names, probabilities), key=lambda item: item[1], reverse=True)
    return [(label, float(score)) for label, score in ranked]
