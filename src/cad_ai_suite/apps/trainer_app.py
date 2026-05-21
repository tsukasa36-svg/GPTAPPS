from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from cad_ai_suite.cad.step_io import summarize_step
from cad_ai_suite.data.label_store import DEFAULT_DB_PATH, list_labels, upsert_label
from cad_ai_suite.ml.train import train_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="CAD AI trainer and labeling app.")
    subparsers = parser.add_subparsers(dest="command")

    label_parser = subparsers.add_parser("label", help="Label a STEP file")
    label_parser.add_argument("path")
    label_parser.add_argument("--label", required=True)
    label_parser.add_argument("--description", default="")
    label_parser.add_argument("--tags", default="")
    label_parser.add_argument("--db", default=str(DEFAULT_DB_PATH))

    list_parser = subparsers.add_parser("list", help="List labels")
    list_parser.add_argument("--db", default=str(DEFAULT_DB_PATH))

    summarize_parser = subparsers.add_parser("summarize", help="Summarize a STEP file")
    summarize_parser.add_argument("path")

    train_parser = subparsers.add_parser("train", help="Train the classifier")
    train_parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    train_parser.add_argument("--out", default="data/models/part_classifier.pt")
    train_parser.add_argument("--epochs", type=int, default=200)

    subparsers.add_parser("ui", help="Open the desktop labeling UI")

    args = parser.parse_args()
    command = args.command or "ui"

    if command == "label":
        record = upsert_label(
            args.path,
            args.label,
            description=args.description,
            tags=args.tags.split(","),
            db_path=args.db,
        )
        print(f"Labeled {record.path} as {record.label}")
    elif command == "list":
        for record in list_labels(args.db):
            tags = ", ".join(record.tags)
            print(f"{record.id}: {record.label} | {record.path} | {tags}")
    elif command == "summarize":
        summary = summarize_step(args.path)
        for key, value in summary.items():
            print(f"{key}: {value}")
    elif command == "train":
        out = train_classifier(args.db, args.out, epochs=args.epochs)
        print(f"Saved model to {out}")
    elif command == "ui":
        TrainerWindow().run()


class TrainerWindow:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("CAD AI Trainer")
        self.root.geometry("820x520")
        self.selected_path = tk.StringVar()
        self.label = tk.StringVar()
        self.tags = tk.StringVar()
        self.description = tk.StringVar()
        self.status = tk.StringVar(value="Choose a STEP file to label.")

        self._build()
        self._refresh_labels()

    def run(self) -> None:
        self.root.mainloop()

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)

        file_row = ttk.Frame(outer)
        file_row.pack(fill="x")
        ttk.Entry(file_row, textvariable=self.selected_path).pack(side="left", fill="x", expand=True)
        ttk.Button(file_row, text="Browse", command=self._browse).pack(side="left", padx=(8, 0))

        form = ttk.LabelFrame(outer, text="Label", padding=12)
        form.pack(fill="x", pady=12)
        ttk.Label(form, text="Part type").grid(row=0, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.label).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Label(form, text="Tags").grid(row=1, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.tags).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Label(form, text="Description").grid(row=2, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.description).grid(row=2, column=1, sticky="ew", padx=8)
        form.columnconfigure(1, weight=1)

        actions = ttk.Frame(outer)
        actions.pack(fill="x")
        ttk.Button(actions, text="Save Label", command=self._save).pack(side="left")
        ttk.Button(actions, text="Train Model", command=self._train).pack(side="left", padx=8)
        ttk.Label(actions, textvariable=self.status).pack(side="left", padx=12)

        self.table = ttk.Treeview(outer, columns=("label", "path", "tags"), show="headings")
        self.table.heading("label", text="Label")
        self.table.heading("path", text="Path")
        self.table.heading("tags", text="Tags")
        self.table.column("label", width=140)
        self.table.column("path", width=440)
        self.table.column("tags", width=180)
        self.table.pack(fill="both", expand=True, pady=(12, 0))

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose STEP file",
            filetypes=[("STEP files", "*.step *.stp"), ("All files", "*.*")],
        )
        if path:
            self.selected_path.set(path)
            summary = summarize_step(path)
            self.status.set(f"{summary['entity_count']} STEP entities found.")

    def _save(self) -> None:
        path = self.selected_path.get().strip()
        label = self.label.get().strip()
        if not path or not label:
            messagebox.showerror("Missing data", "Choose a STEP file and enter a label.")
            return
        upsert_label(
            Path(path),
            label,
            description=self.description.get(),
            tags=self.tags.get().split(","),
        )
        self.status.set("Label saved.")
        self._refresh_labels()

    def _train(self) -> None:
        try:
            out = train_classifier()
        except Exception as exc:
            messagebox.showerror("Training failed", str(exc))
            return
        self.status.set(f"Model saved to {out}")

    def _refresh_labels(self) -> None:
        for item in self.table.get_children():
            self.table.delete(item)
        for record in list_labels():
            self.table.insert(
                "",
                "end",
                values=(record.label, record.path, ", ".join(record.tags)),
            )


if __name__ == "__main__":
    main()
