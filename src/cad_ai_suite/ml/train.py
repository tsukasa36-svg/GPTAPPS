from __future__ import annotations

import argparse
from pathlib import Path

from cad_ai_suite.cad.feature_extraction import FEATURE_KEYS
from cad_ai_suite.data.label_store import DEFAULT_DB_PATH
from cad_ai_suite.ml.dataset import load_training_dataset
from cad_ai_suite.ml.model import build_classifier, require_torch


def train_classifier(
    db_path: str | Path = DEFAULT_DB_PATH,
    out_path: str | Path = "data/models/part_classifier.pt",
    epochs: int = 200,
    learning_rate: float = 0.01,
) -> Path:
    torch, nn = require_torch()
    dataset = load_training_dataset(db_path)
    if len(dataset.label_names) < 1:
        raise ValueError("At least one label class is required.")

    x = torch.tensor(dataset.features, dtype=torch.float32)
    y = torch.tensor(dataset.labels, dtype=torch.long)

    model = build_classifier(
        input_size=len(FEATURE_KEYS),
        output_size=len(dataset.label_names),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_keys": FEATURE_KEYS,
            "label_names": dataset.label_names,
            "means": dataset.means,
            "stds": dataset.stds,
            "epochs": epochs,
            "training_paths": dataset.paths,
        },
        out,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CAD part classifier.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to labels.db")
    parser.add_argument("--out", default="data/models/part_classifier.pt", help="Output model path")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    args = parser.parse_args()

    out = train_classifier(
        db_path=args.db,
        out_path=args.out,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    print(f"Saved model to {out}")


if __name__ == "__main__":
    main()
