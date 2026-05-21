from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from cad_ai_suite.apps.import_steps import DEFAULT_INBOX, import_step_folder
from cad_ai_suite.cad.parametric_builders import export_design, parse_design_prompt
from cad_ai_suite.cad.step_io import summarize_step
from cad_ai_suite.data.label_store import list_labels, upsert_label
from cad_ai_suite.ml.train import train_classifier


RAW_STEP_DIR = Path("data/raw_step")
GENERATED_DIR = Path("data/generated")


class WorkbenchWindow:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("CAD AI Workbench")
        self.root.geometry("1040x720")
        self.root.minsize(920, 620)
        self._configure_style()

        self.import_status = tk.StringVar(value="Choose STEP files or import from step_inbox.")
        self.selected_label_path = tk.StringVar()
        self.label_name = tk.StringVar()
        self.label_tags = tk.StringVar()
        self.label_description = tk.StringVar()
        self.label_status = tk.StringVar(value="Select an imported STEP file to label.")
        self.train_status = tk.StringVar(value="Train after you have labeled imported files.")
        self.prompt = tk.StringVar(
            value="Generate an aluminum L bracket 80mm by 120mm, 6mm thick, with four M6 holes and a triangular gusset"
        )
        self.generate_status = tk.StringVar(value="Enter a part description.")

        self._build()
        self._refresh_all()

    def run(self) -> None:
        self.root.mainloop()

    def _configure_style(self) -> None:
        self.root.configure(bg="#f4f4f5")
        self.root.option_add("*Font", "Helvetica 13")
        self.root.option_add("*Background", "#f4f4f5")
        self.root.option_add("*Foreground", "#111827")
        self.root.option_add("*Entry.Background", "#ffffff")
        self.root.option_add("*Entry.Foreground", "#111827")
        self.root.option_add("*Text.Background", "#ffffff")
        self.root.option_add("*Text.Foreground", "#111827")

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background="#f4f4f5", foreground="#111827", fieldbackground="#ffffff")
        style.configure("TFrame", background="#f4f4f5")
        style.configure("TLabelframe", background="#f4f4f5", foreground="#111827")
        style.configure("TLabelframe.Label", background="#f4f4f5", foreground="#111827")
        style.configure("TLabel", background="#f4f4f5", foreground="#111827")
        style.configure("TButton", background="#e5e7eb", foreground="#111827", padding=6)
        style.map("TButton", background=[("active", "#d1d5db")], foreground=[("active", "#111827")])
        style.configure("TEntry", fieldbackground="#ffffff", foreground="#111827")
        style.configure("TNotebook", background="#f4f4f5", borderwidth=0)
        style.configure("TNotebook.Tab", background="#e5e7eb", foreground="#111827", padding=(14, 8))
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#ffffff"), ("active", "#f9fafb")],
            foreground=[("selected", "#111827"), ("active", "#111827")],
        )
        style.configure(
            "Treeview",
            background="#ffffff",
            foreground="#111827",
            fieldbackground="#ffffff",
            rowheight=28,
        )
        style.configure(
            "Treeview.Heading",
            background="#e5e7eb",
            foreground="#111827",
            font=("Helvetica", 13, "bold"),
        )

    def _build(self) -> None:
        tabs = ttk.Notebook(self.root)
        tabs.pack(fill="both", expand=True)

        self.import_tab = ttk.Frame(tabs, padding=16)
        self.label_tab = ttk.Frame(tabs, padding=16)
        self.train_tab = ttk.Frame(tabs, padding=16)
        self.generate_tab = ttk.Frame(tabs, padding=16)

        tabs.add(self.import_tab, text="Import STEP")
        tabs.add(self.label_tab, text="Label")
        tabs.add(self.train_tab, text="Train")
        tabs.add(self.generate_tab, text="Generate")

        self._build_import_tab()
        self._build_label_tab()
        self._build_train_tab()
        self._build_generate_tab()

    def _build_import_tab(self) -> None:
        actions = ttk.Frame(self.import_tab)
        actions.pack(fill="x")
        ttk.Button(actions, text="Choose STEP Files", command=self._choose_step_files).pack(side="left")
        ttk.Button(actions, text="Import From Inbox", command=self._import_from_inbox).pack(side="left", padx=8)
        ttk.Button(actions, text="Refresh", command=self._refresh_all).pack(side="left")
        ttk.Label(actions, textvariable=self.import_status).pack(side="left", padx=12)

        inbox = ttk.LabelFrame(self.import_tab, text="Drop Folder", padding=12)
        inbox.pack(fill="x", pady=12)
        ttk.Label(inbox, text=str(DEFAULT_INBOX.resolve())).pack(anchor="w")

        self.import_table = ttk.Treeview(
            self.import_tab,
            columns=("file", "entities", "size"),
            show="headings",
            selectmode="browse",
        )
        self.import_table.heading("file", text="Imported File")
        self.import_table.heading("entities", text="Entities")
        self.import_table.heading("size", text="Size")
        self.import_table.column("file", width=680)
        self.import_table.column("entities", width=100, anchor="e")
        self.import_table.column("size", width=120, anchor="e")
        self.import_table.pack(fill="both", expand=True)

    def _build_label_tab(self) -> None:
        chooser = ttk.Frame(self.label_tab)
        chooser.pack(fill="x")
        ttk.Entry(chooser, textvariable=self.selected_label_path).pack(side="left", fill="x", expand=True)
        ttk.Button(chooser, text="Choose Imported STEP", command=self._choose_label_file).pack(side="left", padx=8)

        form = ttk.LabelFrame(self.label_tab, text="Part Label", padding=12)
        form.pack(fill="x", pady=12)
        ttk.Label(form, text="Part type").grid(row=0, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.label_name).grid(row=0, column=1, sticky="ew", padx=8, pady=3)
        ttk.Label(form, text="Tags").grid(row=1, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.label_tags).grid(row=1, column=1, sticky="ew", padx=8, pady=3)
        ttk.Label(form, text="Description").grid(row=2, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.label_description).grid(row=2, column=1, sticky="ew", padx=8, pady=3)
        form.columnconfigure(1, weight=1)

        actions = ttk.Frame(self.label_tab)
        actions.pack(fill="x")
        ttk.Button(actions, text="Save Label", command=self._save_label).pack(side="left")
        ttk.Button(actions, text="Refresh", command=self._refresh_all).pack(side="left", padx=8)
        ttk.Label(actions, textvariable=self.label_status).pack(side="left", padx=12)

        self.label_table = ttk.Treeview(
            self.label_tab,
            columns=("label", "path", "tags"),
            show="headings",
            selectmode="browse",
        )
        self.label_table.heading("label", text="Label")
        self.label_table.heading("path", text="Path")
        self.label_table.heading("tags", text="Tags")
        self.label_table.column("label", width=160)
        self.label_table.column("path", width=560)
        self.label_table.column("tags", width=220)
        self.label_table.pack(fill="both", expand=True, pady=(12, 0))
        self.label_table.bind("<<TreeviewSelect>>", self._on_label_row_selected)

    def _build_train_tab(self) -> None:
        controls = ttk.Frame(self.train_tab)
        controls.pack(fill="x")
        ttk.Button(controls, text="Train Model", command=self._train_model).pack(side="left")
        ttk.Label(controls, textvariable=self.train_status).pack(side="left", padx=12)

        self.train_summary = tk.Text(
            self.train_tab,
            height=24,
            wrap="word",
            bg="#ffffff",
            fg="#111827",
            insertbackground="#111827",
            relief="solid",
            borderwidth=1,
        )
        self.train_summary.pack(fill="both", expand=True, pady=(12, 0))

    def _build_generate_tab(self) -> None:
        ttk.Label(self.generate_tab, text="Prompt").pack(anchor="w")
        ttk.Entry(self.generate_tab, textvariable=self.prompt).pack(fill="x", pady=(4, 8))

        actions = ttk.Frame(self.generate_tab)
        actions.pack(fill="x")
        ttk.Button(actions, text="Generate STEP", command=self._generate_step).pack(side="left")
        ttk.Label(actions, textvariable=self.generate_status).pack(side="left", padx=12)

        self.generate_preview = tk.Text(
            self.generate_tab,
            height=24,
            wrap="word",
            bg="#ffffff",
            fg="#111827",
            insertbackground="#111827",
            relief="solid",
            borderwidth=1,
        )
        self.generate_preview.pack(fill="both", expand=True, pady=(12, 0))

    def _choose_step_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Choose STEP files",
            filetypes=[("STEP files", "*.step *.stp"), ("All files", "*.*")],
        )
        if not paths:
            return
        DEFAULT_INBOX.mkdir(parents=True, exist_ok=True)
        for path in paths:
            source = Path(path)
            destination = _unique_destination(DEFAULT_INBOX / source.name)
            shutil.copy2(source, destination)
        self.import_status.set(f"Copied {len(paths)} file(s) into step_inbox.")
        self._import_from_inbox()

    def _import_from_inbox(self) -> None:
        try:
            records = import_step_folder(move_after_import=True)
        except Exception as exc:
            messagebox.showerror("Import failed", str(exc))
            return
        if records:
            imported_count = sum(record.status == "imported" for record in records)
            self.import_status.set(f"Imported {imported_count} file(s).")
        else:
            self.import_status.set("No STEP/STP files found in step_inbox.")
        self._refresh_all()

    def _choose_label_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose imported STEP file",
            initialdir=str(RAW_STEP_DIR.resolve()),
            filetypes=[("STEP files", "*.step *.stp"), ("All files", "*.*")],
        )
        if path:
            self.selected_label_path.set(path)
            self._summarize_selected_label_file(path)

    def _save_label(self) -> None:
        path = self.selected_label_path.get().strip()
        label = self.label_name.get().strip()
        if not path or not label:
            messagebox.showerror("Missing label", "Choose a STEP file and enter a part type.")
            return
        upsert_label(
            path,
            label,
            description=self.label_description.get(),
            tags=self.label_tags.get().split(","),
        )
        self.label_status.set("Label saved.")
        self._refresh_all()

    def _train_model(self) -> None:
        try:
            out = train_classifier()
        except Exception as exc:
            messagebox.showerror("Training failed", str(exc))
            return
        self.train_status.set(f"Model saved to {out}")
        self._refresh_train_summary()

    def _generate_step(self) -> None:
        prompt = self.prompt.get().strip()
        if not prompt:
            messagebox.showerror("Missing prompt", "Enter a part description.")
            return
        try:
            spec = parse_design_prompt(prompt)
            outputs = export_design(spec, GENERATED_DIR)
        except Exception as exc:
            messagebox.showerror("Generation failed", str(exc))
            return

        self.generate_preview.delete("1.0", "end")
        self.generate_preview.insert("end", json.dumps(spec.to_dict(), indent=2))
        self.generate_preview.insert("end", "\n\nOutputs:\n")
        for key, value in outputs.items():
            self.generate_preview.insert("end", f"{key}: {value}\n")
        self.generate_status.set("Generated CAD output.")

    def _refresh_all(self) -> None:
        self._refresh_import_table()
        self._refresh_label_table()
        self._refresh_train_summary()

    def _refresh_import_table(self) -> None:
        for item in self.import_table.get_children():
            self.import_table.delete(item)
        RAW_STEP_DIR.mkdir(parents=True, exist_ok=True)
        for path in sorted(RAW_STEP_DIR.glob("*")):
            if path.suffix.lower() not in {".step", ".stp"}:
                continue
            try:
                summary = summarize_step(path)
                entities = summary["entity_count"]
                size = summary["file_size_bytes"]
            except Exception:
                entities = "?"
                size = path.stat().st_size
            self.import_table.insert("", "end", values=(str(path), entities, size))

    def _refresh_label_table(self) -> None:
        for item in self.label_table.get_children():
            self.label_table.delete(item)
        for record in list_labels():
            self.label_table.insert(
                "",
                "end",
                values=(record.label, record.path, ", ".join(record.tags)),
            )

    def _refresh_train_summary(self) -> None:
        labels = list_labels()
        counts: dict[str, int] = {}
        for record in labels:
            counts[record.label] = counts.get(record.label, 0) + 1
        lines = [
            f"Labeled files: {len(labels)}",
            "",
            "Label counts:",
        ]
        if counts:
            lines.extend(f"- {label}: {count}" for label, count in sorted(counts.items()))
        else:
            lines.append("- none yet")
        lines.extend(
            [
                "",
                "Model output:",
                str(Path("data/models/part_classifier.pt").resolve()),
            ]
        )
        self.train_summary.delete("1.0", "end")
        self.train_summary.insert("end", "\n".join(lines))

    def _on_label_row_selected(self, _event: tk.Event) -> None:
        selected = self.label_table.selection()
        if not selected:
            return
        values = self.label_table.item(selected[0], "values")
        if len(values) >= 3:
            self.label_name.set(values[0])
            self.selected_label_path.set(values[1])
            self.label_tags.set(values[2])
            self._summarize_selected_label_file(values[1])

    def _summarize_selected_label_file(self, path: str) -> None:
        try:
            summary = summarize_step(path)
        except Exception as exc:
            self.label_status.set(str(exc))
            return
        self.label_status.set(f"{summary['entity_count']} STEP entities found.")


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    index = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def main() -> None:
    WorkbenchWindow().run()


if __name__ == "__main__":
    main()
