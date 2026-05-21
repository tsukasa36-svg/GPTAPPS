from __future__ import annotations

import cgi
import hashlib
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
PORT = int(os.environ.get("CAD_AI_PORT", "8765"))
RAW_STEP_DIR = Path("data/raw_step")
GENERATED_DIR = Path("data/generated")
PREVIEW_DIR = Path("data/previews")


class WebWorkbenchHandler(BaseHTTPRequestHandler):
    server_version = "CADAIWorkbench/0.1"

    def do_GET(self) -> None:
        try:
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
            elif route == "/api/preview-stl":
                query = parse_qs(urlparse(self.path).query)
                path = query.get("path", [""])[0]
                self._send_file(_step_preview_stl(path), "model/stl")
            else:
                self._send_json({"error": "Not found"}, status=404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

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

    def _send_file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _list_imported_files() -> list[dict[str, object]]:
    labels_by_path = {record.path: record for record in list_labels()}
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
        resolved_path = str(path.resolve())
        label = labels_by_path.get(resolved_path)
        files.append(
            {
                "path": resolved_path,
                "name": path.name,
                "size": path.stat().st_size,
                "entity_count": entity_count,
                "schema": schema,
                "label": label.label if label else "",
                "tags": label.tags if label else [],
                "description": label.description if label else "",
            }
        )
    return files


def _step_preview_stl(path: str) -> Path:
    step_path = _resolve_raw_step_path(path)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(
        f"{step_path.resolve()}:{step_path.stat().st_mtime_ns}:{step_path.stat().st_size}".encode("utf-8")
    ).hexdigest()[:20]
    stl_path = PREVIEW_DIR / f"{step_path.stem}_{fingerprint}.stl"
    if stl_path.exists():
        return stl_path

    try:
        import cadquery as cq
    except ImportError as exc:
        raise RuntimeError("CadQuery is required for STEP previews.") from exc

    model = cq.importers.importStep(str(step_path))
    cq.exporters.export(model, str(stl_path))
    return stl_path


def _resolve_raw_step_path(path: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    raw_root = RAW_STEP_DIR.resolve()
    if candidate.suffix.lower() not in {".step", ".stp"}:
        raise ValueError("Preview path must be a STEP/STP file.")
    if raw_root not in candidate.parents:
        raise ValueError("Preview path must be inside data/raw_step.")
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


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
    .label-layout { display: grid; grid-template-columns: minmax(280px, 380px) 1fr; gap: 16px; }
    .file-list { max-height: 520px; overflow: auto; }
    .file-card {
      width: 100%;
      text-align: left;
      background: #fff;
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      margin-bottom: 8px;
      font-weight: 500;
    }
    .file-card:hover, .file-card.selected { background: var(--soft); color: var(--text); border-color: #818cf8; }
    .file-name { display: block; font-weight: 700; overflow-wrap: anywhere; }
    .file-meta { display: block; color: var(--muted); font-size: 13px; margin-top: 4px; }
    .file-label { display: inline-block; margin-top: 6px; padding: 3px 7px; border-radius: 999px; background: #dcfce7; color: #166534; font-size: 12px; }
    .viewer-wrap {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #111827;
      overflow: hidden;
      min-height: 340px;
      position: relative;
    }
    #previewCanvas { width: 100%; height: 360px; display: block; background: #111827; }
    .viewer-note {
      position: absolute;
      left: 12px;
      top: 10px;
      color: #e5e7eb;
      background: rgba(17, 24, 39, 0.72);
      padding: 6px 8px;
      border-radius: 6px;
      font-size: 13px;
    }
    .selected-path {
      padding: 10px;
      background: #f9fafb;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow-wrap: anywhere;
      margin-bottom: 12px;
    }
    @media (max-width: 900px) {
      .grid, .label-layout { grid-template-columns: 1fr; }
    }
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
      <div class="label-layout">
        <div class="panel">
          <h2>Imported STEP Files</h2>
          <p class="muted">Click one file, preview it, then save a label for only that file.</p>
          <div id="labelFileList" class="file-list"></div>
        </div>
        <div class="panel">
          <h2>Selected File</h2>
          <div id="selectedPath" class="selected-path muted">No file selected.</div>
          <div class="viewer-wrap">
            <canvas id="previewCanvas"></canvas>
            <div id="viewerNote" class="viewer-note">Select a STEP file to preview it.</div>
          </div>
          <div class="grid" style="margin-top: 12px;">
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
            <button onclick="saveLabel()">Save Label For Selected File</button>
            <button class="secondary" onclick="refreshAll()">Refresh</button>
          </div>
          <div id="labelStatus" class="status"></div>
        </div>
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
    let selectedPath = "";
    let mesh = null;
    let rotationX = -0.45;
    let rotationY = 0.75;
    let dragging = false;
    let lastPointer = null;

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
      if (selectedPath && !files.some(file => file.path === selectedPath)) selectedPath = "";
      renderFiles();
      renderLabelFileList();
      renderLabels();
      renderTrainingSummary();
      if (!selectedPath && files.length) selectFile(files[0].path, false);
    }

    function renderFiles() {
      const table = document.getElementById("filesTable");
      table.innerHTML = "";
      files.forEach(file => {
        const row = document.createElement("tr");
        const label = file.label ? `${file.label}` : "";
        row.innerHTML = `<td>${escapeHtml(file.name)}</td><td>${file.entity_count ?? ""}</td><td>${file.size}</td><td>${escapeHtml(file.path)} ${label ? `<span class="file-label">${escapeHtml(label)}</span>` : ""}</td>`;
        row.onclick = () => {
          selectFile(file.path, true);
          document.querySelector('[data-tab="label"]').click();
        };
        table.appendChild(row);
      });
    }

    function renderLabelFileList() {
      const list = document.getElementById("labelFileList");
      list.innerHTML = "";
      if (!files.length) {
        list.innerHTML = `<p class="muted">No imported files yet. Use the Import STEP tab first.</p>`;
        return;
      }
      files.forEach(file => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `file-card ${file.path === selectedPath ? "selected" : ""}`;
        const label = file.label ? `<span class="file-label">${escapeHtml(file.label)}</span>` : `<span class="file-meta">Unlabeled</span>`;
        button.innerHTML = `
          <span class="file-name">${escapeHtml(file.name)}</span>
          <span class="file-meta">${file.entity_count ?? "?"} entities | ${file.size} bytes</span>
          ${label}
        `;
        button.onclick = () => selectFile(file.path, true);
        list.appendChild(button);
      });
    }

    function renderLabels() {
      const labelByPath = new Map(labels.map(label => [label.path, label]));
      files.forEach(file => {
        const label = labelByPath.get(file.path);
        file.label = label ? label.label : "";
        file.tags = label ? label.tags : [];
        file.description = label ? label.description : "";
      });
    }

    async function selectFile(path, loadViewer = true) {
      selectedPath = path;
      const file = files.find(item => item.path === path);
      document.getElementById("selectedPath").textContent = file ? `${file.name} - ${file.path}` : path;
      document.getElementById("partLabel").value = file?.label || "";
      document.getElementById("partTags").value = file?.tags?.join(", ") || "";
      document.getElementById("partDescription").value = file?.description || "";
      document.getElementById("labelStatus").textContent = file ? `Selected ${file.name}. Saving will update only this file.` : "Selected file.";
      renderLabelFileList();
      if (loadViewer) await loadPreview(path);
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
      const path = selectedPath;
      const label = document.getElementById("partLabel").value.trim();
      if (!path || !label) {
        document.getElementById("labelStatus").textContent = "Click one imported file and enter a part type.";
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
      const file = files.find(item => item.path === path);
      document.getElementById("labelStatus").textContent = `Label saved for ${file ? file.name : path}.`;
      await refreshAll();
      selectFile(path, false);
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

    async function loadPreview(path) {
      const note = document.getElementById("viewerNote");
      note.textContent = "Loading 3D preview...";
      mesh = null;
      drawPreview();
      try {
        const response = await fetch(`/api/preview-stl?path=${encodeURIComponent(path)}`);
        if (!response.ok) {
          const text = await response.text();
          throw new Error(text);
        }
        const buffer = await response.arrayBuffer();
        mesh = parseBinaryStl(buffer);
        note.textContent = `${mesh.triangles.length} preview triangles. Drag to rotate.`;
        drawPreview();
      } catch (error) {
        note.textContent = `Preview unavailable: ${error.message}`;
      }
    }

    function parseBinaryStl(buffer) {
      const view = new DataView(buffer);
      const triangleCount = view.getUint32(80, true);
      const triangles = [];
      const points = [];
      let offset = 84;
      for (let i = 0; i < triangleCount && offset + 50 <= view.byteLength; i++) {
        const normal = [
          view.getFloat32(offset, true),
          view.getFloat32(offset + 4, true),
          view.getFloat32(offset + 8, true)
        ];
        offset += 12;
        const vertices = [];
        for (let j = 0; j < 3; j++) {
          const vertex = [
            view.getFloat32(offset, true),
            view.getFloat32(offset + 4, true),
            view.getFloat32(offset + 8, true)
          ];
          offset += 12;
          vertices.push(vertex);
          points.push(vertex);
        }
        offset += 2;
        triangles.push({ normal, vertices });
      }
      const bounds = computeBounds(points);
      return { triangles, bounds };
    }

    function computeBounds(points) {
      const min = [Infinity, Infinity, Infinity];
      const max = [-Infinity, -Infinity, -Infinity];
      points.forEach(point => {
        for (let i = 0; i < 3; i++) {
          min[i] = Math.min(min[i], point[i]);
          max[i] = Math.max(max[i], point[i]);
        }
      });
      const center = min.map((value, index) => (value + max[index]) / 2);
      const size = Math.max(...max.map((value, index) => value - min[index]), 1);
      return { min, max, center, size };
    }

    function setupCanvas() {
      const canvas = document.getElementById("previewCanvas");
      canvas.addEventListener("pointerdown", event => {
        dragging = true;
        lastPointer = [event.clientX, event.clientY];
        canvas.setPointerCapture(event.pointerId);
      });
      canvas.addEventListener("pointermove", event => {
        if (!dragging || !lastPointer) return;
        const dx = event.clientX - lastPointer[0];
        const dy = event.clientY - lastPointer[1];
        rotationY += dx * 0.01;
        rotationX += dy * 0.01;
        lastPointer = [event.clientX, event.clientY];
        drawPreview();
      });
      canvas.addEventListener("pointerup", () => {
        dragging = false;
        lastPointer = null;
      });
      window.addEventListener("resize", drawPreview);
      drawPreview();
    }

    function drawPreview() {
      const canvas = document.getElementById("previewCanvas");
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);
      ctx.fillStyle = "#111827";
      ctx.fillRect(0, 0, rect.width, rect.height);
      drawGrid(ctx, rect.width, rect.height);
      if (!mesh) return;

      const scale = Math.min(rect.width, rect.height) * 0.72 / mesh.bounds.size;
      const projected = mesh.triangles.map(triangle => {
        const vertices = triangle.vertices.map(vertex => projectVertex(vertex, mesh.bounds.center, scale, rect.width, rect.height));
        const depth = vertices.reduce((sum, vertex) => sum + vertex.z, 0) / 3;
        const shade = Math.max(60, Math.min(220, 130 + triangle.normal[2] * 60 + depth * 0.03));
        return { vertices, depth, shade };
      }).sort((a, b) => a.depth - b.depth);

      projected.forEach(triangle => {
        ctx.beginPath();
        ctx.moveTo(triangle.vertices[0].x, triangle.vertices[0].y);
        ctx.lineTo(triangle.vertices[1].x, triangle.vertices[1].y);
        ctx.lineTo(triangle.vertices[2].x, triangle.vertices[2].y);
        ctx.closePath();
        ctx.fillStyle = `rgb(${triangle.shade}, ${Math.round(triangle.shade * 0.95)}, ${Math.round(triangle.shade * 0.82)})`;
        ctx.strokeStyle = "rgba(255,255,255,0.12)";
        ctx.lineWidth = 0.7;
        ctx.fill();
        ctx.stroke();
      });
    }

    function drawGrid(ctx, width, height) {
      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      ctx.lineWidth = 1;
      for (let x = 0; x < width; x += 40) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y < height; y += 40) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }
    }

    function projectVertex(vertex, center, scale, width, height) {
      let x = vertex[0] - center[0];
      let y = vertex[1] - center[1];
      let z = vertex[2] - center[2];
      const cosY = Math.cos(rotationY);
      const sinY = Math.sin(rotationY);
      const x1 = x * cosY + z * sinY;
      const z1 = -x * sinY + z * cosY;
      const cosX = Math.cos(rotationX);
      const sinX = Math.sin(rotationX);
      const y1 = y * cosX - z1 * sinX;
      const z2 = y * sinX + z1 * cosX;
      return {
        x: width / 2 + x1 * scale,
        y: height / 2 - y1 * scale,
        z: z2
      };
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    setupCanvas();
    refreshAll();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
