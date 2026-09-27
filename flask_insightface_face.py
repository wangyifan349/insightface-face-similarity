"""
Flask UI for InsightFace (buffalo_l) face similarity. Two panels, all AJAX.

    1. 查询图 vs 候选图片     one query image against every image uploaded in
                            the same request. The local folder is not touched.
    2. 查询图 vs 人脸库目录   one query image against every image inside a
                            folder on disk (1:N).

The face library is any folder of images. It comes from the
INSIGHTFACE_FACE_LIBRARY_DIR environment variable, or from the library_dir form
field, otherwise face_library/ next to this file is used.

Scores are cosine similarity percentages of 512-d ArcFace embeddings, so they
are NOT comparable with the dlib version's percentages. Two photos of the same
person usually land around 45-70, two different people around 0-40. Nothing
here decides whether two faces belong to the same person; pick your own
threshold from your own samples.

Run:

    python flask_insightface_face.py
    python flask_insightface_face.py --port 5100 --dir D:\\faces
"""

from __future__ import annotations

import argparse
import os
import socket
import threading
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template_string, request

from insightface_face_similarity import (
    DESCRIPTOR_DIMENSION,
    MODEL_NAME,
    encode_faces,
    similarity_matrix_percent,
    warm_up,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024          # 64 MB per request

LIBRARY_DIRECTORY_VARIABLE = "INSIGHTFACE_FACE_LIBRARY_DIR"
PORT_ENVIRONMENT_VARIABLE = "PORT"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_TOP_RESULTS = 10
SCRIPT_DIRECTORY = Path(__file__).resolve().parent

# (path, modified time) -> list of embeddings, so the library is encoded once.
_LIBRARY_CACHE: dict = {}
_LIBRARY_LOCK = threading.Lock()


# -------------------- Face Library --------------------
def default_library_directory() -> Path:
    """The library folder used when the request does not name one."""
    configured = os.environ.get(LIBRARY_DIRECTORY_VARIABLE)
    if configured:
        return Path(configured).expanduser()
    return SCRIPT_DIRECTORY / "face_library"


def resolve_library_directory(requested: str = "") -> Path:
    """Pick the library folder for this request.

    A folder typed into the form wins over the environment variable, so the
    page can search any folder without restarting the server.
    """
    cleaned = (requested or "").strip().strip('"').strip("'")
    if not cleaned:
        return default_library_directory()
    expanded = os.path.expandvars(os.path.expanduser(cleaned))
    directory = Path(expanded)
    if not directory.is_dir():
        raise ValueError(f"人脸库目录不存在: {directory}")
    return directory


def library_images(directory: Path) -> list:
    """Every image under a folder, recursively, in a stable order."""
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_library(directory: Path) -> tuple:
    """Return (labels, matrix) for every face in the library folder.

    Labels are file names, with " #2", " #3" appended when one photo holds
    several faces. Embeddings are cached per file and invalidated by the file's
    modified time, so editing or deleting an image is picked up automatically.
    """
    paths = library_images(directory)

    labels: list = []
    rows: list = []
    seen: set = set()

    with _LIBRARY_LOCK:
        for index, path in enumerate(paths, start=1):
            if len(paths) >= 20:
                print(f"  载入人脸库 {index}/{len(paths)}  {path.name}", flush=True)

            key = str(path)
            try:
                modified_time = path.stat().st_mtime
            except OSError:
                # Deleted between the scan and now; nothing to encode.
                continue
            cached = _LIBRARY_CACHE.get(key)

            if cached is None or cached[0] != modified_time:
                try:
                    embeddings = encode_faces(path)
                except (OSError, ValueError):
                    embeddings = []
                _LIBRARY_CACHE[key] = (modified_time, embeddings)
            else:
                embeddings = cached[1]

            seen.add(key)
            for position, embedding in enumerate(embeddings):
                label = path.name if position == 0 else f"{path.name} #{position + 1}"
                labels.append(label)
                rows.append(embedding)

        for stale_key in [key for key in _LIBRARY_CACHE if key not in seen]:
            del _LIBRARY_CACHE[stale_key]

    if not rows:
        return labels, np.zeros((0, DESCRIPTOR_DIMENSION), dtype="float32")
    return labels, np.vstack(rows).astype("float32")


def search_matrix(query_embeddings: list, library_matrix: np.ndarray) -> np.ndarray:
    """Best cosine score for every library face, in percent.

    The query may hold several faces (a group photo), so each library face is
    scored against the best matching query face.
    """
    return similarity_matrix_percent(query_embeddings, [row for row in library_matrix])


def bar_range(scores: list) -> tuple:
    """Window the similarity bars are drawn against.

    On the ArcFace scale, real scores cluster in a narrow band, so a fixed
    0-100 bar would make every row look identical. When the rows differ enough
    to tell apart, the bars are scaled to this run's own lowest and highest
    score and the range is reported next to the table.
    """
    if len(scores) < 2:
        return 0.0, 100.0
    low, high = min(scores), max(scores)
    if high - low < 0.5:
        return 0.0, 100.0
    return low, high


def rank_scores(labels: list, scores: np.ndarray, top_results: int, kind: str) -> list:
    """Turn raw scores into a ranked, rounded list."""
    order = np.argsort(-scores)[: max(1, int(top_results))]
    matches = [
        {
            "rank": rank,
            "kind": kind,
            "candidate": labels[int(index)],
            "similarity": round(float(scores[int(index)]), 2),
        }
        for rank, index in enumerate(order, start=1)
    ]
    if matches:
        low, high = bar_range([row["similarity"] for row in matches])
        for row in matches:
            row["bar_low"] = round(low, 2)
            row["bar_high"] = round(high, 2)
    return matches


def disambiguate(names: list) -> list:
    """Append an index to repeated names so the result stays readable."""
    totals: dict = {}
    for name in names:
        totals[name] = totals.get(name, 0) + 1
    seen: dict = {}
    labels = []
    for name in names:
        if totals[name] == 1:
            labels.append(name)
            continue
        seen[name] = seen.get(name, 0) + 1
        labels.append(f"{name} ({seen[name]})")
    return labels


# -------------------- Request Helpers --------------------
def read_uploads(field_name: str) -> list:
    """Return every uploaded file of a field as a (filename, bytes) pair."""
    return [
        (item.filename, item.read())
        for item in request.files.getlist(field_name)
        if item and item.filename
    ]


def encode_upload(data: bytes):
    """Return the embeddings of one image, or an error message."""
    try:
        embeddings = encode_faces(data)
    except (OSError, ValueError) as error:
        return f"处理失败：{error}"
    if not embeddings:
        return "未检测到人脸"
    return embeddings


def top_results_from_request() -> int:
    raw_value = request.form.get("top_results") or request.args.get("top_results")
    try:
        return max(1, min(100, int(raw_value)))
    except (TypeError, ValueError):
        return DEFAULT_TOP_RESULTS


def requested_library_directory() -> Path:
    return resolve_library_directory(
        request.form.get("library_dir") or request.args.get("library_dir") or ""
    )


def failure(message: str, status: int = 400):
    return jsonify({"ok": False, "error": message}), status


def library_failure(error: Exception) -> tuple:
    """Turn a bad library folder into a readable 400 instead of a 500."""
    return failure(f"人脸库不可用：{error}")


# -------------------- Page --------------------
PAGE = """
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>InsightFace 人脸相似度</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
      rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"
        defer></script>
<style>
  :root {
    --brand: #e8590c;
    --brand-dark: #d9480f;
    --brand-tint: #fff4e6;
    /* Bootstrap paints links with the *rgb triplet* variables, not with
       --bs-link-color, so both forms have to be redirected or <a> stays
       #0d6efd blue. */
    --bs-link-color: #d9480f;
    --bs-link-hover-color: #c2410c;
    --bs-link-color-rgb: 217, 72, 15;
    --bs-link-hover-color-rgb: 194, 65, 12;
    /* Everything else in Bootstrap that ships as blue (#0d6efd / #cfe2ff). */
    --bs-primary-rgb: 232, 89, 12;
    --bs-focus-ring-color: rgba(232, 89, 12, .25);
    --bs-table-accent-bg: #fff4e6;
    --bs-table-active-bg: #fff4e6;
    --bs-secondary-bg: #f6f3f0;
    --bs-tertiary-bg: #f6f3f0;
    --bs-secondary-color: #9a8577;
  }
  /* Belt and braces: state the link colours directly so no cascade order or
     later Bootstrap rule can leave an anchor blue. */
  a, a:link, a:visited, a:hover, a:focus, a:active {
    color: #d9480f;
  }
  a:hover, a:focus { color: #c2410c; }
  /* The browser paints ::selection blue by default. */
  ::selection { background: #f3d6c0; color: #7a2e0a; }
  /* Every focusable element, in case Bootstrap's ring is overridden upstream. */
  a:focus-visible, .btn:focus-visible, .btn-close:focus-visible,
  .form-control:focus-visible, .form-select:focus-visible,
  input:focus-visible, textarea:focus-visible, [tabindex]:focus-visible {
    outline: none;
    box-shadow: 0 0 0 .25rem rgba(232, 89, 12, .25);
  }
  /* The file picker button is a real control on this page; it is blue by
     default in Bootstrap. */
  input[type="file"]::file-selector-button {
    background: #fff4e6; color: #7a2e0a;
    border: 1px solid #f3d6c0; border-radius: 6px;
    padding: .35rem .8rem; margin-right: .75rem;
  }
  input[type="file"]::file-selector-button:hover {
    background: #f3d6c0; border-color: #e8590c; color: #7a2e0a;
  }
  /* Unused on this page, overridden anyway so no future markup can leak blue. */
  .form-check-input:checked {
    background-color: #e8590c; border-color: #e8590c;
  }
  .form-check-input:focus {
    border-color: #e8590c; box-shadow: 0 0 0 .25rem rgba(232, 89, 12, .25);
  }
  .alert-link, .page-link { color: #c2410c; }
  html, body { height: 100%; }
  body {
    background: #f6f3f0;
    font-family: system-ui, "Microsoft YaHei", "Segoe UI", sans-serif;
  }
  /* Keep every accent orange-red: no blue anywhere. */
  .btn { --bs-btn-focus-shadow-rgb: 232, 89, 12; }
  .btn-primary {
    --bs-btn-color: #fff;
    --bs-btn-bg: #e8590c;
    --bs-btn-border-color: #e8590c;
    --bs-btn-hover-color: #fff;
    --bs-btn-hover-bg: #d9480f;
    --bs-btn-hover-border-color: #d9480f;
    --bs-btn-active-color: #fff;
    --bs-btn-active-bg: #c2410c;
    --bs-btn-active-border-color: #c2410c;
    --bs-btn-disabled-color: #fff;
    --bs-btn-disabled-bg: #f3a06a;
    --bs-btn-disabled-border-color: #f3a06a;
    --bs-btn-focus-shadow-rgb: 232, 89, 12;
  }
  .panel {
    background: #fff;
    border: 1px solid #ece4dc;
    border-top: 4px solid var(--brand);
    border-radius: 10px;
    box-shadow: 0 1px 3px rgba(60, 30, 10, .06);
    height: 100%;
  }
  .panel-body { padding: 1.5rem 1.5rem 1.75rem; }
  .panel-title {
    font-size: 1.02rem; font-weight: 700; color: #7a2e0a;
    display: flex; align-items: center; gap: .55rem;
  }
  .step {
    width: 1.75rem; height: 1.75rem; border-radius: 50%;
    background: var(--brand); color: #fff;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: .9rem; font-weight: 700; flex: none;
  }
  .form-control:focus, .form-select:focus {
    border-color: var(--brand);
    box-shadow: 0 0 0 .25rem rgba(232, 89, 12, .22);
  }
  .form-text { color: #9a8577; }
  .btn-lg { padding: .7rem 1.4rem; font-size: 1rem; font-weight: 600; }
  .table thead th {
    background: var(--brand-tint); color: #7a2e0a;
    border-bottom: 2px solid #f3d6c0; font-size: .82rem;
    text-transform: uppercase; letter-spacing: .04em;
  }
  .table td { vertical-align: middle; }
  .badge-score {
    background: var(--brand); color: #fff;
    font-size: .95rem; font-weight: 700; min-width: 5.2rem;
  }
  .bar { height: 6px; border-radius: 3px; background: #f1e2d6; overflow: hidden; }
  .bar > span { display: block; height: 100%; background: var(--brand); }
  .alert-brand { background: var(--brand-tint); border: 1px solid #f3c9a8; color: #7a2e0a; }
  .empty-state { color: #a8968a; font-size: .92rem; }
  .thumb { max-height: 88px; border-radius: 6px; border: 1px solid #e6dbd1; }
  .spinner {
    width: 1rem; height: 1rem; border: 2px solid rgba(255,255,255,.45);
    border-top-color: #fff; border-radius: 50%;
    display: inline-block; animation: spin .7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .verdict {
    background: var(--brand-tint); border: 1px solid #f3c9a8;
    border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 1.25rem;
  }
  .verdict .who { font-size: 1.3rem; font-weight: 700; color: #7a2e0a; }
  .model-note {
    display: flex; flex-wrap: wrap; align-items: center; gap: .45rem .6rem;
    background: #fff; border: 1px solid #ece4dc; border-radius: 8px;
    padding: .6rem .9rem; font-size: .86rem; color: #8a6a52;
  }
  .model-note .chip {
    background: #fff4e6; border: 1px solid #f3d6c0; color: #7a2e0a;
    border-radius: 4px; padding: .1rem .45rem; font-size: .78rem;
  }
  .model-note .mono {
    font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
    color: #7a2e0a;
  }
  .model-note .sep {
    width: 1px; height: 1rem; background: #ece4dc;
  }
</style>
</head>
<body>
<div class="container-fluid px-4 px-lg-5 py-4">

  <div class="d-flex flex-wrap align-items-end justify-content-between gap-3 mb-3">
    <div>
      <h1 class="h3 mb-1" style="color:#7a2e0a">InsightFace 人脸相似度</h1>
      <div class="text-secondary">余弦相似度 · {{ model }} · {{ dimension }} 维 ArcFace 向量</div>
    </div>
    <div class="text-end">
      <span class="badge rounded-pill text-bg-light border" id="libraryBadge">人脸库载入中…</span>
      <div class="form-text mt-1" id="libraryPath"></div>
    </div>
  </div>

  <div class="model-note mb-4">
    <span class="chip">内核</span>
    <span class="mono">{{ model }}</span>
    <span class="sep"></span>
    <span class="chip">检测</span>
    <span class="mono">SCRFD det_10g</span>
    <span class="sep"></span>
    <span class="chip">识别</span>
    <span class="mono">ArcFace w600k_r50</span>
    <span class="sep"></span>
    <span class="chip">向量</span>
    <span class="mono">{{ dimension }} 维</span>
    <span class="sep"></span>
    <span class="chip">推理</span>
    <span class="mono">onnxruntime</span>
  </div>

  <div class="row g-4 align-items-stretch">
    <div class="col-12 col-xl-6">
      <div class="panel">
        <div class="panel-body">
          <div class="panel-title mb-1"><span class="step">1</span>查询图 vs 候选图片（1:N）</div>
          <p class="form-text mb-3">一张查询图，逐个和下面选中的候选图比对，<strong>不使用</strong>本地人脸库。</p>
          <form id="queryForm" novalidate>
            <div class="mb-3">
              <label class="form-label fw-semibold" for="queryFile">① 查询图片（单张）</label>
              <input class="form-control form-control-lg" type="file" id="queryFile"
                     name="query" accept="image/*" required>
              <div class="form-text" data-summary="queryFile">例如：查询这是谁.jpg</div>
              <div class="d-flex flex-wrap gap-2 mt-2" data-preview="queryFile"></div>
            </div>
            <div class="mb-3">
              <label class="form-label fw-semibold" for="candidateFiles">② 候选图片（可多选）</label>
              <input class="form-control form-control-lg" type="file" id="candidateFiles"
                     name="candidates" accept="image/*" multiple required>
              <div class="form-text" data-summary="candidateFiles">例如：1用户1.jpg、2用户2.jpg</div>
              <div class="d-flex flex-wrap gap-2 mt-2" data-preview="candidateFiles"></div>
            </div>
            <div class="mb-4" style="max-width: 12rem">
              <label class="form-label fw-semibold" for="queryTop">返回条数</label>
              <input class="form-control form-control-lg" type="number" id="queryTop"
                     name="top_results" value="{{ top_results }}" min="1" max="100">
            </div>
            <button class="btn btn-primary btn-lg w-100" type="submit" id="querySubmit">
              开始比对
            </button>
          </form>
        </div>
      </div>
    </div>

    <div class="col-12 col-xl-6">
      <div class="panel">
        <div class="panel-body">
          <div class="panel-title mb-1"><span class="step">2</span>查询图 vs 人脸库目录（1:N）</div>
          <p class="form-text mb-3">一张查询图，对比指定文件夹里的<b>全部</b>图片，递归扫描子目录。</p>
          <form id="libraryForm" novalidate>
            <div class="mb-3">
              <label class="form-label fw-semibold" for="libraryQuery">① 查询图片（单张）</label>
              <input class="form-control form-control-lg" type="file" id="libraryQuery"
                     name="query" accept="image/*" required>
              <div class="form-text" data-summary="libraryQuery">例如：查询这是谁.jpg</div>
              <div class="d-flex flex-wrap gap-2 mt-2" data-preview="libraryQuery"></div>
            </div>
            <div class="mb-3">
              <label class="form-label fw-semibold" for="libraryDir">② 人脸库文件夹</label>
              <input class="form-control form-control-lg" type="text" id="libraryDir"
                     name="library_dir" value="{{ library_dir }}" placeholder="留空则用默认 face_library">
              <div class="form-text">可以粘贴绝对路径，识别 .jpg .jpeg .png .bmp .webp</div>
            </div>
            <div class="mb-4" style="max-width: 12rem">
              <label class="form-label fw-semibold" for="libraryTop">返回条数</label>
              <input class="form-control form-control-lg" type="number" id="libraryTop"
                     name="top_results" value="{{ top_results }}" min="1" max="100">
            </div>
            <button class="btn btn-primary btn-lg w-100" type="submit" id="librarySubmit">
              搜索人脸库
            </button>
          </form>
        </div>
      </div>
    </div>
  </div>

  <div class="row g-4 mt-0">
    <div class="col-12">
      <div class="panel">
        <div class="panel-body">
          <div class="d-flex flex-wrap align-items-center justify-content-between gap-2 mb-3">
            <div class="panel-title mb-0"><span class="step">3</span>结果</div>
            <div class="form-text" id="resultMeta"></div>
          </div>
          <div id="resultAlert"></div>
          <div id="resultBody" class="empty-state">尚未发起比对。</div>
        </div>
      </div>
    </div>
  </div>

</div>

<script>
const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value == null ? "" : String(value);
  return div.innerHTML;
}

function setLoading(button, loading, idleText) {
  button.disabled = loading;
  button.innerHTML = loading
    ? '<span class="spinner me-2"></span>比对中…'
    : idleText;
}

function showError(message) {
  $("resultAlert").innerHTML =
    '<div class="alert alert-brand alert-dismissible fade show" role="alert">' +
    '<strong>出错了：</strong>' + escapeHtml(message) +
    '<button type="button" class="btn-close" data-bs-dismiss="alert"></button></div>';
  $("resultBody").innerHTML = "";
  $("resultMeta").textContent = "";
}

function resultTable(rows) {
  if (!rows || !rows.length) {
    return '<p class="empty-state mb-0">没有可比较的对象。</p>';
  }
  const scaled = rows[0].bar_high - rows[0].bar_low > 0.5;
  const body = rows.map((row) => {
    const span = scaled
      ? Math.max(0, Math.min(100, (row.similarity - row.bar_low)
          / (row.bar_high - row.bar_low) * 100))
      : Math.max(0, Math.min(100, row.similarity));
    return `
    <tr${row.best ? ' class="table-warning"' : ''}>
      <td style="width:4rem">${row.rank}</td>
      <td>${escapeHtml(row.candidate)}</td>
      <td style="width:9rem">
        <span class="badge badge-score">${row.similarity.toFixed(2)}%</span>
      </td>
      <td style="width:32%">
        <div class="bar"><span style="width:${span}%"></span></div>
      </td>
    </tr>`;
  }).join("");
  const note = scaled
    ? `<p class="form-text mt-2 mb-0">分布条按本次 ${rows[0].bar_low.toFixed(2)}% -
       ${rows[0].bar_high.toFixed(2)}% 缩放，左边数值是真实相似度。</p>`
    : "";
  return `<table class="table table-sm align-middle mb-0">
    <thead><tr><th>排名</th><th>${escapeHtml(rows[0].kind || "候选")}</th><th>相似度</th><th></th></tr></thead>
    <tbody>${body}</tbody></table>` + note;
}

function skippedList(skipped) {
  if (!skipped || !skipped.length) return "";
  return '<p class="form-text mt-3 mb-0">跳过：'
    + skipped.map((item) => escapeHtml(item.name) + "（" + escapeHtml(item.status) + "）").join("、")
    + '</p>';
}

function verdictBlock(matches) {
  if (!matches || !matches.length) {
    return '<p class="empty-state">没有可用候选。</p>';
  }
  const best = matches[0];
  return `<div class="verdict d-flex flex-wrap align-items-center justify-content-between gap-3">
      <div>
        <div class="form-text mb-1">最像的是</div>
        <div class="who">${escapeHtml(best.candidate)}</div>
      </div>
      <div class="text-end">
        <div class="form-text mb-1">相似度</div>
        <div style="font-size:2.4rem;color:#7a2e0a">${best.similarity.toFixed(2)}%</div>
      </div>
    </div>`;
}

function renderSingle(payload) {
  $("resultBody").innerHTML =
    verdictBlock(payload.matches) + resultTable(payload.matches)
    + skippedList(payload.skipped);
}

function renderPayload(payload) {
  $("resultAlert").innerHTML = "";
  const meta = [];
  if (payload.query) meta.push("查询：" + payload.query);
  if (payload.elapsed_ms != null) meta.push("总耗时 " + payload.elapsed_ms + " ms");
  if (payload.count != null) meta.push("已处理 " + payload.count + " 张");
  if (payload.library_faces != null) meta.push("人脸库 " + payload.library_faces + " 张脸");
  if (payload.library_images != null) meta.push("目录 " + payload.library_images + " 张图");
  $("resultMeta").textContent = meta.join(" · ");

  renderSingle(payload);
}

async function postJson(url, formData, button, idleText) {
  setLoading(button, true, idleText);
  $("resultAlert").innerHTML = "";
  try {
    const response = await fetch(url, { method: "POST", body: formData });
    let payload;
    try {
      payload = await response.json();
    } catch (parseError) {
      throw new Error("服务器返回了无法解析的内容（HTTP " + response.status + "）");
    }
    if (!response.ok || payload.ok === false) {
      throw new Error(payload.error || ("请求失败（HTTP " + response.status + "）"));
    }
    renderPayload(payload);
  } catch (error) {
    showError(error.message);
  } finally {
    setLoading(button, false, idleText);
  }
}

function bindForm(formId, buttonId, url, idleText, requiredIds) {
  $(formId).addEventListener("submit", (event) => {
    event.preventDefault();
    const form = event.target;
    for (const id of requiredIds) {
      if (!form.querySelector("#" + id).files.length) {
        showError("请先选择这个面板需要的图片");
        return;
      }
    }
    postJson(url, new FormData(form), $(buttonId), idleText);
  });
}

bindForm("queryForm", "querySubmit", "/api/query-set", "开始比对",
         ["queryFile", "candidateFiles"]);
bindForm("libraryForm", "librarySubmit", "/api/query-library", "搜索人脸库",
         ["libraryQuery"]);

document.querySelectorAll('input[type="file"]').forEach((input) => {
  input.addEventListener("change", () => {
    const picked = Array.from(input.files);
    const summary = document.querySelector('[data-summary="' + input.id + '"]');
    if (summary && input.multiple) {
      summary.textContent = picked.length
        ? "已选择 " + picked.length + " 张图片"
        : "尚未选择";
    } else if (summary && picked.length) {
      summary.textContent = "已选择：" + picked[0].name;
    }
    const preview = document.querySelector('[data-preview="' + input.id + '"]');
    if (!preview) return;
    const tiles = picked.slice(0, 8).map((file) =>
      '<img class="thumb" src="' + URL.createObjectURL(file) + '" title="'
      + escapeHtml(file.name) + '">');
    if (picked.length > 8) {
      tiles.push('<span class="thumb d-flex align-items-center px-2 text-secondary">+'
        + (picked.length - 8) + '</span>');
    }
    preview.innerHTML = tiles.join("");
  });
});

function loadLibraryStatus(directory) {
  const suffix = directory ? "?dir=" + encodeURIComponent(directory) : "";
  return fetch("/api/library" + suffix)
    .then((response) => response.json())
    .then((payload) => {
      if (!payload.ok) {
        $("libraryBadge").textContent = "人脸库不可用";
        $("libraryPath").textContent = payload.error || "";
        return;
      }
      $("libraryBadge").textContent =
        "人脸库 " + payload.faces + " 张人脸 / " + payload.images + " 张图";
      $("libraryPath").textContent = payload.path;
    })
    .catch(() => { $("libraryBadge").textContent = "人脸库状态未知"; });
}

loadLibraryStatus("");

document.getElementById("libraryDir").addEventListener("change", (event) => {
  loadLibraryStatus(event.target.value.trim());
});
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(
        PAGE,
        model=MODEL_NAME,
        dimension=DESCRIPTOR_DIMENSION,
        top_results=DEFAULT_TOP_RESULTS,
        library_dir=str(default_library_directory()),
    )


@app.get("/api/library")
def api_library():
    """Face library status: how many images and how many faces inside."""
    try:
        directory = resolve_library_directory(request.args.get("dir", ""))
    except ValueError as error:
        return failure(str(error))

    labels, matrix = load_library(directory)
    return jsonify(
        {
            "ok": True,
            "path": str(directory),
            "images": len(library_images(directory)),
            "faces": int(matrix.shape[0]),
        }
    )


@app.post("/api/query-set")
def api_query_set():
    """Panel 1: one query image against every image uploaded in this request."""
    started = time.perf_counter()
    query_upload = read_uploads("query")
    if not query_upload:
        return failure("请选择一张查询图片")
    if len(query_upload) > 1:
        return failure("查询图片只能选一张")

    candidate_uploads = read_uploads("candidates")
    if not candidate_uploads:
        return failure("请至少选择一张候选图片")

    query_name, query_data = query_upload[0]
    query_embeddings = encode_upload(query_data)
    if isinstance(query_embeddings, str):
        return failure(f"查询图片{query_embeddings}：" + query_name)

    labels = []
    candidate_rows = []
    skipped = []
    for name, data in candidate_uploads:
        embeddings = encode_upload(data)
        if isinstance(embeddings, str):
            skipped.append({"name": name, "status": embeddings})
            continue
        candidate_rows.append(embeddings[0])
        labels.append(name)

    if not candidate_rows:
        return failure("候选图片里没有可用的人脸")

    labels = disambiguate(labels)
    scores = similarity_matrix_percent(query_embeddings, candidate_rows)
    matches = rank_scores(labels, scores, top_results_from_request(), "候选图片")
    matches[0]["best"] = True

    return jsonify(
        {
            "ok": True,
            "mode": "query-set",
            "query": query_name,
            "query_faces": len(query_embeddings),
            "count": len(candidate_uploads),
            "compared": len(candidate_rows),
            "matches": matches,
            "skipped": skipped,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
    )


@app.post("/api/query-library")
def api_query_library():
    """Panel 2: one query image against every image in a folder on disk."""
    started = time.perf_counter()
    query_upload = read_uploads("query")
    if not query_upload:
        return failure("请选择一张查询图片")
    if len(query_upload) > 1:
        return failure("查询图片只能选一张")

    try:
        directory = requested_library_directory()
    except ValueError as error:
        return library_failure(error)

    query_name, query_data = query_upload[0]
    query_embeddings = encode_upload(query_data)
    if isinstance(query_embeddings, str):
        return failure(f"查询图片{query_embeddings}：" + query_name)

    images = library_images(directory)
    if not images:
        return failure(f"人脸库里没有图片：{directory}")

    library_labels, matrix = load_library(directory)
    if matrix.shape[0] == 0:
        return failure(f"人脸库里没有检测到人脸：{directory}")

    matches = rank_scores(
        library_labels,
        search_matrix(query_embeddings, matrix),
        top_results_from_request(),
        "库中候选",
    )
    matches[0]["best"] = True

    return jsonify(
        {
            "ok": True,
            "mode": "query-library",
            "query": query_name,
            "query_faces": len(query_embeddings),
            "library": str(directory),
            "library_images": len(images),
            "library_faces": int(matrix.shape[0]),
            "matches": matches,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
    )


@app.errorhandler(413)
def too_large(_error):
    return failure("上传文件太大，单次请求上限 64 MB", 413)


def find_free_port(preferred_port: int, attempts: int = 20) -> int:
    """Return the first free port, so a busy 5000 does not break startup."""
    for candidate in range(preferred_port, preferred_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            # No SO_REUSEADDR here: on Windows it would let the probe bind a
            # port another process is already listening on.
            try:
                probe.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            return candidate
    raise RuntimeError(
        f"No free port between {preferred_port} and {preferred_port + attempts - 1}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="InsightFace 人脸相似度网页版：1:N 上传比对、1:N 目录比对。"
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get(PORT_ENVIRONMENT_VARIABLE, "5000")),
        help="起始端口，被占用时自动顺延，默认 5000",
    )
    parser.add_argument(
        "--dir", default="", help="人脸库文件夹，默认用脚本同级的 face_library"
    )
    parser.add_argument(
        "--library-info",
        action="store_true",
        help="只统计人脸库有多少图多少人脸然后退出，不启动网页",
    )
    arguments = parser.parse_args(argv)

    if arguments.dir:
        os.environ[LIBRARY_DIRECTORY_VARIABLE] = arguments.dir

    if arguments.library_info:
        directory = default_library_directory()
        images = library_images(directory)
        print(f"正在加载 {MODEL_NAME} 模型，约 2 秒…", flush=True)
        print(f"模型加载完成，用时 {warm_up():.0f} ms", flush=True)
        print(f"人脸库目录: {directory}")
        if not images:
            print("人脸库里没有找到图片（.jpg .jpeg .png .bmp .webp）")
            return 0
        _labels, matrix = load_library(directory)
        print(f"图片 {len(images)} 张，检测到人脸 {matrix.shape[0]} 张")
        return 0

    print(f"正在加载 {MODEL_NAME} 模型，约 2 秒…", flush=True)
    print(f"模型加载完成，用时 {warm_up():.0f} ms", flush=True)
    print(f"人脸库目录: {default_library_directory()}", flush=True)

    listening_port = find_free_port(arguments.port)
    print(f"Open http://127.0.0.1:{listening_port} in your browser", flush=True)
    app.run(host="127.0.0.1", port=listening_port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
