from __future__ import annotations

import cgi
import json
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
import webbrowser

from cad_ai_suite.apps.import_steps import DEFAULT_INBOX, import_step_folder
from cad_ai_suite.cad.parametric_builders import export_design, parse_design_prompt
from cad_ai_suite.cad.step_io import summarize_step
from cad_ai_suite.data.label_store import list_labels, upsert_label
from cad_ai_suite.ml.train import train_classifier


HOST = "127.0.0.1"
PORT = 8765
RAW_STEP_DIR = Path("data/raw_step")
GENERATED_DIR = Path("data/generated")


class WebWorkbenchHandler(BaseHTTPRequestHandler):
    server_version = "CADAIWorkbench/0.1"

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/":
            self._send_html(INDEX_HTML)
        elif route == "/api/files":
            self._send_json({"files": _list_imported_files()})
        elif route == "/api/labels":
            self._send_json({"labels": [_label_to_dict(record) for record in list_labels()]})
        elif route == "/api/summary":
            query = parse_qs(urlparse(self.path).query)
            path = query.get("path", [""])[0]
            self._send_json({"summary": summarize_step(path)})
        else:
            self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        try:
            if route == "/api/upload":
                self._handle_upload()
            elif route == "/api/import-inbox":
                records = import_step_folder(move_after_import=True)
                self._send_json({"records": [record.__dict__ for record in records]})
            elif route == "/api/label":
                payload = self._read_json()
                record = upsert_label(
                    payload["path"],
                    payload["label"],
                    description=payload.get("description", ""),
                    tags=str(payload.get("tags", "")).split(","),
                )
                self._send_json({"label": _label_to_dict(record)})
            elif route == "/api/train":
                out = train_classifier()
                self._send_json({"model": str(out)})
            elif route == "/api/generate":
                payload = self._read_json()
                spec = parse_design_prompt(payload["prompt"])
                outputs = export_design(spec, GENERATED_DIR)
                self._send_json(
                    {
                        "spec": spec.to_dict(),
                        "outputs": {key: str(value) for key, value in outputs.items()},
                    }
                )
            else:
                self._send_json({"error": "Not found"}, status=404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _handle_upload(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            self._send_json({"error": "Expected multipart form upload"}, status=400)
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
            },
        )
        files = form["files"] if "files" in form else []
        if not isinstance(files, list):
            files = [files]

        DEFAULT_INBOX.mkdir(parents=True, exist_ok=True)
        saved = []
        for item in files:
            if not getattr(item, "filename", None):
                continue
            filename = Path(item.filename).name
            if Path(filename).suffix.lower() not in {".step", ".stp"}:
                continue
            destination = _unique_destination(DEFAULT_INBOX / filename)
            data = item.file.read()
            destination.write_bytes(data)
            saved.append(str(destination))

        records = import_step_folder(move_after_import=True)
        self._send_json(
            {
                "saved": saved,
                "records": [record.__dict__ for record in records],
            }
        )

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length)
        return json.loads(data.decode("utf-8"))

    def _send_html(self, html: str) -> None:
        encoded = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: dict[str, object], status: int = 200) -> None:
        encoded = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def _list_imported_files() -> list[dict[str, object]]:
    RAW_STEP_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for path in sorted(RAW_STEP_DIR.glob("*")):
        if path.suffix.lower() not in {".step", ".stp"}:
            continue
        try:
            summary = summarize_step(path)
            entity_count = summary["entity_count"]
            schema = summary["schema"]
        except Exception:
            entity_count = None
            schema = None
        files.append(
            {
                "path": str(path.resolve()),
                "name": path.name,
                "size": path.stat().st_size,
                "entity_count": entity_count,
                "schema": schema,
            }
        )
    return files


def _label_to_dict(record) -> dict[str, object]:
    return {
        "id": record.id,
        "path": record.path,
        "label": record.label,
        "description": record.description,
        "tags": record.tags,
    }


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
    os.chdir(Path(__file__).resolve().parents[3])
    server = ThreadingHTTPServer((HOST, PORT), WebWorkbenchHandler)
    url = f"http://{HOST}:{PORT}"
    print(f"CAD AI Workbench running at {url}")
    webbrowser.open(url)
    server.serve_forever()


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CAD AI Workbench</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f4f5;
      --panel: #ffffff;
      --text: #111827;
      --muted: #6b7280;
      --line: #d1d5db;
      --button: #2563eb;
      --button-dark: #1d4ed8;
      --soft: #eef2ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      padding: 18px 24px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    h1 { margin: 0; font-size: 22px; }
    main { padding: 20px 24px 40px; max-width: 1180px; margin: 0 auto; }
    .tabs { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
    .tab {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      padding: 10px 14px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 14px;
    }
    .tab.active { background: var(--soft); border-color: #818cf8; }
    section { display: none; }
    section.active { display: block; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
    }
    .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    label { display: block; font-weight: 600; margin-bottom: 6px; }
    input[type="text"], select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      color: var(--text);
      background: #fff;
      font: inherit;
    }
    textarea { min-height: 180px; resize: vertical; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    button {
      border: 0;
      background: var(--button);
      color: #fff;
      padding: 10px 14px;
      border-radius: 6px;
      cursor: pointer;
      font: inherit;
      font-weight: 600;
    }
    button.secondary { background: #4b5563; }
    button:hover { background: var(--button-dark); }
    table { width: 100%; border-collapse: collapse; background: #fff; }
    th, td { text-align: left; border-bottom: 1px solid var(--line); padding: 10px; font-size: 14px; }
    th { background: #f9fafb; }
    tr { cursor: pointer; }
    tr.selected { background: var(--soft); }
    .muted { color: var(--muted); }
    .status { min-height: 24px; color: #065f46; font-weight: 600; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    @media (max-width: 760px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>CAD AI Workbench</h1>
    <span class="muted">Local browser GUI</span>
  </header>
  <main>
    <div class="tabs">
      <button class="tab active" data-tab="import">Import STEP</button>
      <button class="tab" data-tab="label">Label</button>
      <button class="tab" data-tab="train">Train</button>
      <button class="tab" data-tab="generate">Generate</button>
    </div>

    <section id="import" class="active">
      <div class="panel">
        <label>Choose STEP/STP files</label>
        <div class="row">
          <input id="fileInput" type="file" accept=".step,.stp" multiple>
          <button onclick="uploadFiles()">Import Selected Files</button>
          <button class="secondary" onclick="importInbox()">Import From step_inbox</button>
          <button class="secondary" onclick="refreshAll()">Refresh</button>
        </div>
        <p class="muted">You can also drop files into the project folder's step_inbox directory, then click Import From step_inbox.</p>
        <div id="importStatus" class="status"></div>
      </div>
      <div class="panel">
        <h2>Imported Files</h2>
        <table>
          <thead><tr><th>Name</th><th>Entities</th><th>Size</th><th>Path</th></tr></thead>
          <tbody id="filesTable"></tbody>
        </table>
      </div>
    </section>

    <section id="label">
      <div class="panel">
        <div class="grid">
          <div>
            <label>Imported file</label>
            <select id="labelPath"></select>
          </div>
          <div>
            <label>Part type</label>
            <input id="partLabel" type="text" placeholder="l_bracket, fixture, mounting_plate">
          </div>
          <div>
            <label>Tags</label>
            <input id="partTags" type="text" placeholder="bracket, mounting, holes">
          </div>
          <div>
            <label>Description</label>
            <input id="partDescription" type="text" placeholder="Optional notes">
          </div>
        </div>
        <div class="row" style="margin-top: 12px;">
          <button onclick="saveLabel()">Save Label</button>
          <button class="secondary" onclick="refreshAll()">Refresh</button>
        </div>
        <div id="labelStatus" class="status"></div>
      </div>
      <div class="panel">
        <h2>Labels</h2>
        <table>
          <thead><tr><th>Label</th><th>Tags</th><th>Path</th></tr></thead>
          <tbody id="labelsTable"></tbody>
        </table>
      </div>
    </section>

    <section id="train">
      <div class="panel">
        <button onclick="trainModel()">Train Model</button>
        <div id="trainStatus" class="status"></div>
      </div>
      <div class="panel">
        <h2>Training Summary</h2>
        <pre id="trainSummary"></pre>
      </div>
    </section>

    <section id="generate">
      <div class="panel">
        <label>Prompt</label>
        <input id="prompt" type="text" value="Generate an aluminum L bracket 80mm by 120mm, 6mm thick, with four M6 holes and a triangular gusset">
        <div class="row" style="margin-top: 12px;">
          <button onclick="generatePart()">Generate STEP</button>
        </div>
        <div id="generateStatus" class="status"></div>
      </div>
      <div class="panel">
        <h2>Generated Output</h2>
        <textarea id="generateOutput" readonly></textarea>
      </div>
    </section>
  </main>

  <script>
    let files = [];
    let labels = [];

    document.querySelectorAll(".tab").forEach(button => {
      button.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach(tab => tab.classList.remove("active"));
        document.querySelectorAll("section").forEach(section => section.classList.remove("active"));
        button.classList.add("active");
        document.getElementById(button.dataset.tab).classList.add("active");
      });
    });

    async function api(path, options = {}) {
      const response = await fetch(path, options);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Request failed");
      return data;
    }

    async function refreshAll() {
      const filesData = await api("/api/files");
      const labelsData = await api("/api/labels");
      files = filesData.files;
      labels = labelsData.labels;
      renderFiles();
      renderLabels();
      renderTrainingSummary();
    }

    function renderFiles() {
      const table = document.getElementById("filesTable");
      const select = document.getElementById("labelPath");
      table.innerHTML = "";
      select.innerHTML = "";
      files.forEach(file => {
        const row = document.createElement("tr");
        row.innerHTML = `<td>${file.name}</td><td>${file.entity_count ?? ""}</td><td>${file.size}</td><td>${file.path}</td>`;
        row.onclick = () => {
          document.getElementById("labelPath").value = file.path;
          document.querySelector('[data-tab="label"]').click();
        };
        table.appendChild(row);
        const option = document.createElement("option");
        option.value = file.path;
        option.textContent = file.name;
        select.appendChild(option);
      });
    }

    function renderLabels() {
      const table = document.getElementById("labelsTable");
      table.innerHTML = "";
      labels.forEach(label => {
        const row = document.createElement("tr");
        row.innerHTML = `<td>${label.label}</td><td>${label.tags.join(", ")}</td><td>${label.path}</td>`;
        row.onclick = () => {
          document.getElementById("labelPath").value = label.path;
          document.getElementById("partLabel").value = label.label;
          document.getElementById("partTags").value = label.tags.join(", ");
          document.getElementById("partDescription").value = label.description || "";
        };
        table.appendChild(row);
      });
    }

    function renderTrainingSummary() {
      const counts = {};
      labels.forEach(label => counts[label.label] = (counts[label.label] || 0) + 1);
      const lines = [`Labeled files: ${labels.length}`, "", "Label counts:"];
      const keys = Object.keys(counts).sort();
      if (keys.length === 0) lines.push("- none yet");
      keys.forEach(key => lines.push(`- ${key}: ${counts[key]}`));
      lines.push("", "Model output:", "data/models/part_classifier.pt");
      document.getElementById("trainSummary").textContent = lines.join("\n");
    }

    async function uploadFiles() {
      const input = document.getElementById("fileInput");
      if (!input.files.length) {
        document.getElementById("importStatus").textContent = "Choose at least one STEP/STP file.";
        return;
      }
      const form = new FormData();
      for (const file of input.files) form.append("files", file);
      document.getElementById("importStatus").textContent = "Importing...";
      const data = await api("/api/upload", { method: "POST", body: form });
      document.getElementById("importStatus").textContent = `Imported ${data.records.length} file(s).`;
      await refreshAll();
    }

    async function importInbox() {
      document.getElementById("importStatus").textContent = "Importing from step_inbox...";
      const data = await api("/api/import-inbox", { method: "POST" });
      document.getElementById("importStatus").textContent = data.records.length ? `Imported ${data.records.length} file(s).` : "No files found.";
      await refreshAll();
    }

    async function saveLabel() {
      const path = document.getElementById("labelPath").value;
      const label = document.getElementById("partLabel").value.trim();
      if (!path || !label) {
        document.getElementById("labelStatus").textContent = "Choose a file and enter a part type.";
        return;
      }
      await api("/api/label", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          path,
          label,
          tags: document.getElementById("partTags").value,
          description: document.getElementById("partDescription").value
        })
      });
      document.getElementById("labelStatus").textContent = "Label saved.";
      await refreshAll();
    }

    async function trainModel() {
      document.getElementById("trainStatus").textContent = "Training...";
      try {
        const data = await api("/api/train", { method: "POST" });
        document.getElementById("trainStatus").textContent = `Saved model to ${data.model}`;
      } catch (error) {
        document.getElementById("trainStatus").textContent = error.message;
      }
    }

    async function generatePart() {
      const prompt = document.getElementById("prompt").value.trim();
      document.getElementById("generateStatus").textContent = "Generating...";
      try {
        const data = await api("/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt })
        });
        document.getElementById("generateOutput").value = JSON.stringify(data, null, 2);
        document.getElementById("generateStatus").textContent = "Generated CAD output.";
      } catch (error) {
        document.getElementById("generateStatus").textContent = error.message;
      }
    }

    refreshAll();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
