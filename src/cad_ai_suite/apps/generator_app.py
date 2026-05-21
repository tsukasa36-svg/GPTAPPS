from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

import tkinter as tk
from tkinter import messagebox, ttk

from cad_ai_suite.cad.parametric_builders import export_design, parse_design_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate parametric CAD from text.")
    parser.add_argument("prompt", nargs="*", help="Text prompt describing the part")
    parser.add_argument("--out-dir", default="data/generated")
    parser.add_argument("--ui", action="store_true", help="Open the desktop generator UI")
    args = parser.parse_args()

    if args.ui or not args.prompt:
        GeneratorWindow(output_dir=args.out_dir).run()
        return

    prompt = " ".join(args.prompt)
    spec = parse_design_prompt(prompt)
    outputs = export_design(spec, args.out_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


class GeneratorWindow:
    def __init__(self, output_dir: str | Path = "data/generated") -> None:
        self.output_dir = Path(output_dir)
        self.root = tk.Tk()
        self.root.title("CAD AI Generator")
        self.root.geometry("820x560")
        self.prompt = tk.StringVar(
            value="Generate an aluminum L bracket 80mm by 120mm, 6mm thick, with four M6 holes and a triangular gusset"
        )
        self.status = tk.StringVar(value="Enter a part description.")
        self._build()

    def run(self) -> None:
        self.root.mainloop()

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="Prompt").pack(anchor="w")
        prompt_entry = ttk.Entry(outer, textvariable=self.prompt)
        prompt_entry.pack(fill="x", pady=(4, 8))

        actions = ttk.Frame(outer)
        actions.pack(fill="x")
        ttk.Button(actions, text="Generate", command=self._generate).pack(side="left")
        ttk.Label(actions, textvariable=self.status).pack(side="left", padx=12)

        self.preview = tk.Text(outer, height=22, wrap="word")
        self.preview.pack(fill="both", expand=True, pady=(12, 0))

    def _generate(self) -> None:
        prompt = self.prompt.get().strip()
        if not prompt:
            messagebox.showerror("Missing prompt", "Enter a part description.")
            return
        spec = parse_design_prompt(prompt)
        outputs = export_design(spec, self.output_dir)
        self.preview.delete("1.0", "end")
        self.preview.insert("end", json.dumps(spec.to_dict(), indent=2))
        self.preview.insert("end", "\n\nOutputs:\n")
        for key, value in outputs.items():
            self.preview.insert("end", f"{key}: {value}\n")
        self.status.set("Generated design spec.")


if __name__ == "__main__":
    main()
