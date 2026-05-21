from __future__ import annotations


def require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for this command. Install it with: "
            'pip install -e ".[ml]"'
        ) from exc
    return torch, nn


def build_classifier(input_size: int, output_size: int, hidden_size: int = 64):
    torch, nn = require_torch()
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )
