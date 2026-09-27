# 🎯 InsightFace Face Similarity / Face Search (buffalo_l · ArcFace)

<p align="center">
  <a href="README.md"><strong>🇨🇳 中文</strong></a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="README_EN.md"><strong>🇬🇧 English</strong></a>
</p>

A thin wrapper on top of the official [InsightFace](https://github.com/deepinsight/insightface) `buffalo_l` model bundle:
it detects the faces in an image, extracts 512-dimensional feature vectors, and then computes a **cosine similarity percentage**.

- 🆔 **1:1** compare two images once
- 🔎 **1:N** rank one query image against a batch of candidates (several uploaded images, or a folder face library)
- 🧩 Three ways to use it: **web UI** (Flask + AJAX), **command line / double-click**, and **import as a Python library**
- ⚡ **Runs on CPU only**, and automatically uses CUDA (when there is an N card and `onnxruntime-gpu` is installed)
- 🔒 All photos are processed **locally**; nothing is ever uploaded to a third-party server

> ⚠️ **This tool only outputs a similarity score. It does NOT decide whether two photos are the same person.**
> It gives you a number between 0 and 100; whether you base a decision on it, and which threshold you use, is your own call.
> Run it once with **your own** samples (several of the same person and several of different people), look at the score distribution, and only then set a threshold.

---

## 📖Table of Contents

- [✨ Highlights](#highlights)
- [📦 Repository contents](#repository-contents)
- [🧠 Model: what it actually uses](#model-what-it-actually-uses)
- [🔢 Algorithm: from an image to a percentage](#algorithm-from-an-image-to-a-percentage)
- [❓ Why scores are 45~70 and not 99](#why-scores-are-4570-and-not-99)
- [💡 An easy pitfall: the smaller the face in the image, the lower the score](#an-easy-pitfall-the-smaller-the-face-in-the-image-the-lower-the-score)
- [🔍 Face selection rules](#face-selection-rules)
- [🚀 Quick start](#quick-start)
- [🖥 Way 1: web UI](#way-1-web-ui)
- [⌨ Way 2: command line](#way-2-command-line)
- [📚 Way 3: use as a library](#way-3-use-as-a-library)
- [🔌 HTTP API](#http-api)
- [🔧 Environment variables](#environment-variables)
- [📊 Measured performance](#measured-performance)
- [🐳 Deployment](#deployment)
- [🆘 FAQ](#faq)
- [🙏 Acknowledgements and citations](#acknowledgements-and-citations)
- [📜 License](#license)

---

## ✨Highlights

| Item | Description |
|---|---|
| Core | InsightFace `buffalo_l` (SCRFD-10GF detection + ArcFace ResNet50@WebFace600K recognition) |
| Inference | onnxruntime (CPU / CUDA); no PyTorch needed, nothing to compile |
| Features | 512-dimensional ArcFace vectors, already L2-normalized |
| Scoring | cosine similarity of the two vectors × 100 |
| Dependencies | Python 3.10+, insightface, onnxruntime, opencv-python, numpy (the web UI additionally needs Flask) |
| Model | the official `buffalo_l` is downloaded automatically on first run (about 326 MB); only 2 of its files are loaded (182 MB) |
| Network | only the first model download needs it; after that it runs fully offline |

---

## 📦Repository contents

```
insightface-face-similarity/
├── insightface_face_similarity.py  core library: detection + features + similarity (the only file that really does the work)
├── example_insightface.py          minimal example, 6 lines, demonstrates 1:1 only
├── example2_insightface.py         command line + double-click interactive mode, 1:1 and 1:N
├── flask_insightface_face.py       web UI, two panels, both 1:N
├── 使用说明_insightface.txt         complete Chinese API documentation (all functions, parameters and return structures listed)
├── README.md                       this file (Chinese)
└── README_EN.md                    English version, same content
```

The four files are independent of each other; the only shared dependency is `insightface_face_similarity.py`,
so it has to sit in the same directory as the other three (or be on `PYTHONPATH`).

```
insightface_face_similarity.py
        ^
        | depended on
        |
  +-----+-----------+----------------+
  |                 |                |
example_insightface.py  example2_insightface.py  flask_insightface_face.py
   minimal example     command line / double-click          web UI
```

---

## 🧠Model: what it actually uses

InsightFace's official `buffalo_l` model bundle is always used. This bundle contains 5 `.onnx` files in total,
and **this tool only loads 2 of them**:

| File | Purpose | Size | Loaded? |
|---|---|---|---|
| `det_10g.onnx` | SCRFD-10GF face detection, outputs detection boxes + 5 keypoints | 16.1 MB | ✅ |
| `w600k_r50.onnx` | ArcFace (ResNet50@WebFace600K) face recognition, outputs a 512-dimensional vector | 166.3 MB | ✅ |
| `1k3d68.onnx` | 3D landmark model | 137 MB | ❌ skipped |
| `2d106det.onnx` | 106-point 2D landmark model | 4.8 MB | ❌ skipped |
| `genderage.onnx` | gender / age model | 1.3 MB | ❌ skipped |

Skipping the last three is **deliberate**: computing a similarity needs neither landmarks nor age.
Measured on this machine (same machine, CPU only, two consecutive runs, the stable value taken):

| | Load time |
|---|---|
| Load all 5 models | 1.52 ~ 1.69 s |
| **Load only 2 models (this tool)** | **1.12 ~ 1.17 s** |

So the bundle is about 326 MB on disk, but only 182 MB of weights actually enter memory (143 MB of 3D landmark weights are not loaded).
At the code level this tool does it through `allowed_modules=["detection", "recognition"]`,
see `insightface_face_similarity.py:196`.

### 📊 Official buffalo_l recognition accuracy

The table below comes from the [official InsightFace Model Zoo](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md),
and it holds the model's evaluation results on the standard datasets (not measured by this tool):

| Metric | MR-ALL | African | Caucasian | South Asian | East Asian | LFW | CFP-FP | AgeDB-30 | IJB-C(E4) |
|---|---|---|---|---|---|---|---|---|---|
| buffalo_l | 91.25 | 90.29 | 94.70 | 93.16 | 74.96 | 99.83 | 99.33 | 98.23 | 97.25 |

Worth noting: the East Asian column is clearly low (74.96), which shows that the model does not perform evenly across different groups of people.

### 🆚 Compared with other face libraries: who is more accurate

Conclusion first, data afterwards.

**Conclusion**: among solutions of the "detection + feature + cosine" kind, the ArcFace family is in the first tier;
dlib and OpenCV SFace, lightweight solutions of that sort, are a few percentage points behind but much faster;
FaceNet512 is the highest on the same leaderboard, with ArcFace about 1.8 points behind.
**This project uses ArcFace, not FaceNet**.

#### ✅ The only directly comparable table

The table below comes from the [official DeepFace benchmark](https://github.com/serengil/deepface/tree/master/benchmarks):
**the same LFW dataset, the same verification pairs, the same cosine distance, alignment enabled everywhere, and RetinaFace as the detector for every row**.
The protocol is identical, so comparing the models horizontally within this table is valid.

| Recognition model | LFW accuracy | Feature dimension | Notes |
|---|---|---|---|
| FaceNet512 | **98.4** | 512 | highest in this table |
| ArcFace | **96.6** | 512 | **used by this project** |
| FaceNet | 96.4 | 128 | strongest model with the smallest dimension in this table |
| VGG-Face | 95.8 | 4096 | big features, slow inference |
| SFace | 92.4 | 128 | MobileFaceNet class, extremely lightweight |
| GhostFaceNet | 90.5 | 128 | goes for extreme lightness |
| Dlib | 89.1 | 128 | see "How big is the gap" below |
| OpenFace | 69.4 | 128 | superseded by the generations above it |
| DeepFace | 67.7 | 4096 | an older model |
| DeepID | 67.7 | 160 | an even earlier model |
| *Human* | *97.5* | — | control baseline on the same batch of data |

#### 📏 How big is the gap

Under an **exactly identical protocol**:

- **ArcFace 96.6% vs Dlib 89.1% → a gap of 7.5 percentage points**.
  Converted to an error rate: 3.4% vs 10.9%, so dlib's error rate is about **3.2 times** ArcFace's.
- ArcFace is still 0.9 points behind the human baseline (97.5%).
- **The detector matters no less than the model**: with ArcFace unchanged,
  swapping the detector from RetinaFace to OpenCV Haar drops it to 84.6%, while "no detection, use the original image directly" gives only 54.8%.
  So don't stare only at the recognition model, `det_10g.onnx` (SCRFD) is just as important.
- The price: ArcFace(R50) weights are 166 MB, 512-dimensional, and clearly slower on CPU than dlib(128-dimensional small network).
  If you want speed, switch to SFace/GhostFaceNet and accept losing a few percentage points.

#### 📋 Self-reported numbers from each project (different protocols, not directly comparable)

| Project | Self-reported LFW | Source |
|---|---|---|
| InsightFace `buffalo_l` (this project) | 99.83 | official Model Zoo |
| InsightFace ArcFace R50 | 99.65+ | [arcface repo](https://github.com/chenggongliang/arcface) |
| OpenCV SFace | 99.40 | [opencv_zoo model card](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/README.md) |
| dlib `face_recognition_ex` | 99.38 | [official dlib example](https://dlib.net/dnn_face_recognition_ex.cpp.html) |

> ⚠️ These four numbers **must not be used to rank anything**. They each use a different verification subset, threshold search method and
> preprocessing pipeline, so their absolute values are generally higher than in the table above. Cross-table comparison only holds "within the same table".
>
> Also note: the **percentage this tool outputs is not an accuracy**. It is the cosine similarity between the two images right now,
> see "Why scores are 45~70 and not 99".

---

## 🔢Algorithm: from an image to a percentage

```
Image (path / bytes / numpy array)
   │
   ├─ 1. Decode into a BGR uint8 array
   │      Paths containing non-ASCII characters are read fine too: internally it is Path.read_bytes() + cv2.imdecode,
   │      not cv2.imread (it cannot open non-ASCII paths on Windows)
   │
   ├─ 2. SCRFD detection (fixed 640x640 input, confidence threshold 0.5)
   │      Output: a number of detection boxes + 5 keypoints for each face
   │      All faces in an image are kept; if there is not a single face an empty result is returned (no exception raised)
   │
   ├─ 3. Use the detector's 5 keypoints + the ArcFace standard template for affine alignment, crop to 112x112
   │      This step happens inside insightface (face_align.norm_crop),
   │      so this tool does not need the 2d106det model——it uses the detector's own 5 points,
   │      not that 106-point model
   │
   ├─ 4. ArcFace forward pass → 512-dimensional vector → L2 normalization (norm = 1.0)
   │      The recognition model's input is 112x112, normalized as (pixel - 127.5) / 127.5, output 512-dimensional
   │      after one more L2 normalization the dot product equals the cosine similarity
   │
   └─ 5. Similarity = dot(a, b) × 100
```

Code location: `insightface_face_similarity.py:272` (`encode_faces`),
`insightface_face_similarity.py:326` (`cosine_similarity_percent`).

### ⚡ Why 1:N is fast

Once every candidate image has had its features computed, **one matrix multiplication** produces all the results, with no Python loop:

```python
scores = (候选矩阵 @ 查询矩阵.T).max(axis=1) * 100
```

`insightface_face_similarity.py:354` (`similarity_matrix_percent`).
`.max(axis=1)` means "one candidate vs all faces in the query image, take the highest pair",
so **passing a group photo as the query image still correctly finds the most similar person**.

### 🔀 The only difference from the default InsightFace behaviour

In insightface 2.0, `app.prepare()` defaults to `det_size=None` (Auto),
and internally runs detection at 128x128 and 640x640 and then merges the results.
This tool **fixes it at 640x640 only**, and there is exactly one reason for doing that:
**to make the score reproducible** —— the same person on the same two images always gets the same score.
The price is that small faces fare slightly worse (see the next section).

---

## ❓Why scores are 45~70 and not 99

Because the **raw cosine values between 512-dimensional ArcFace vectors simply are not close to 1**.
This is normal in deep face recognition: what the model optimizes for is "angle separability", not "angle normalization".

Reference values measured on this machine (buffalo_l, CPU only):

| Scenario | Similarity |
|---|---|
| The same image compared with itself | **100.00%** |
| The same image, after the detector has scaled it (assembled into a large group photo and compared) | 95 ~ 97% |
| The same person, different angle / expression / lighting | 50 ~ 70 |
| Two different people | 38 ~ 50 |
| Two different people (fairly different) | 10 ~ 30 |

So:

- **Two different people can also reach 50%** (look-alikes, similar faces, kinship); a high score ≠ the same person;
- **The same person can also be only 50%**; a low score ≠ different people;
- There is no universal threshold; you can only calibrate it with your own data.

> As a side note: if you have also seen dlib-version 128-dimensional scores ("above 85% for the same person"),
> those two percentages **cannot be compared with each other**—they are two completely different rulers. The threshold has to be recalibrated.

---

## 💡An easy pitfall: the smaller the face in the image, the lower the score

The detector always looks at the whole image at 640x640. A large 4000x3000 group photo gets scaled down to a width of 640,
so the faces inside are reduced to a few dozen pixels, and the extracted features are not that stable:
**the same face viewed on its own at full size scores 70%, but stuffed into a large group photo it may be only 50%.**

- Scores are only comparable when the images in the face library **have a resolution close to the query image**;
- Searching 4K images with a thumbnail of a few hundred pixels drops the hit rate noticeably;
- When this happens, **cropping the image to around the face** before comparing is much more accurate.

---

## 🔍Face selection rules

One image can contain several faces, and the face selection rules are not exactly the same in every part of the code, so they are listed publicly to avoid misunderstandings:

| Entry point | Query image | Candidates / face library |
|---|---|---|
| Web panel ① (`/api/query-set`) | **all faces** take part, the highest pair wins | **only the largest face is taken** |
| Web panel ② (`/api/query-library`) | **all faces** take part | **one candidate per face** (when one file holds several faces they are labelled `filename #2`, `#3`) |
| `example2_insightface.py` (default) | **all faces** take part | **all faces** take part, the highest pair wins |
| `example2_insightface.py --largest-only` | only the largest face | only the largest face |
| `face_similarity_percent()` (core library) | by default only the largest face | by default only the largest face; add `compare_all_faces=True` to compare arbitrary pairs |

The command line and the web UI go through two different scoring paths (the command line uses `best_pair_percent`,
the web UI uses `similarity_matrix_percent`), but the **face selection** step is the same function,
`encode_faces`; it guarantees that both paths get the same set of vectors, sorted by area,
so there is no face selection difference of the kind "the web UI finds it but the command line does not".

---

## 🚀Quick start

### 1. Requirements

- **Python 3.10 or newer** (a requirement since insightface 2.0)
- Windows / Linux / macOS
- Installed directly into the current environment, no virtual environment needed
- The first run needs network access to download the models; after that it runs fully offline

### 2. Install

```bash
git clone https://github.com/wangyifan349/insightface-face-similarity.git
cd insightface-face-similarity

python -m pip install -U pip
```

> All the commands below are run in the **current environment**. Using `python -m pip` instead of bare `pip`,
> is precisely to make sure the package lands in this very Python (on Windows `pip` sometimes points at a different interpreter).

#### One-liner to detect an NVIDIA GPU

You can paste this single line straight into PowerShell / cmd / bash (the output is ASCII, to avoid Windows console encoding problems):

```bash
python -c "import shutil,subprocess; g=shutil.which('nvidia-smi') and subprocess.run(['nvidia-smi'],capture_output=True).returncode==0; print('[GPU ] NVIDIA GPU found -> install onnxruntime-gpu' if g else '[CPU ] no NVIDIA GPU found -> install onnxruntime')"
```

If it prints `[GPU ]`, take the GPU line below; if it prints `[CPU ]`, take the CPU line.

#### Conditional install based on the detection result

Same single line: if a GPU is detected, install the GPU build, otherwise the CPU build:

```bash
python -c "import shutil,subprocess,sys; g=shutil.which('nvidia-smi') and subprocess.run(['nvidia-smi'],capture_output=True).returncode==0; p=(['insightface','onnxruntime-gpu'] if g else ['insightface','onnxruntime'])+['opencv-python','numpy','flask']; print('pip install -U '+' '.join(p)); sys.exit(subprocess.call([sys.executable,'-m','pip','install','-U']+p))"
```

If you want to see every step, or to reproduce it elsewhere, write it out separately:

```bash
# has an N card
python -m pip install insightface onnxruntime-gpu opencv-python numpy flask

# no N card
python -m pip install insightface onnxruntime opencv-python numpy flask
```

`insightface` itself pulls in `onnxruntime`, `opencv-python` and `numpy`,
so they are all spelled out above only to make the intent clear and avoid pitfalls.

> ⚠️ `nvidia-smi` running successfully **does not mean** the GPU build will work.
> `onnxruntime-gpu` also has to match the local CUDA / cuDNN version, otherwise it silently falls back to CPU.
> See the `info` command in the "GPU deployment" section for how to verify this.
>
> macOS has no `nvidia-smi`, so it always takes the CPU dependencies (CoreML can be specified manually).

### 3. Verify the install

```bash
python insightface_face_similarity.py info
```

Output is similar to:

```
model=buffalo_l dim=512 providers=not loaded yet det=640x640@0.5 models=C:\Users\<you>\.insightface\models\buffalo_l
```

- `providers` is the inference backend actually in use; since you have just run `info` the models are not loaded yet, so it shows `not loaded yet`,
  and only after a real comparison has been made once (or with `python -c "import insightface_face_similarity as m; m.warm_up(); print(m.describe_configuration())"`) does it show the real backend
- `models` is where the model files live
- If it says a download is needed, the first run will fetch about 326 MB of `buffalo_l`

### 4. Run it

```bash
# fastest way to verify
python insightface_face_similarity.py a.jpg b.jpg

# web UI
python flask_insightface_face.py
```

---

## 🖥Way 1: web UI

```bash
python flask_insightface_face.py                     # default face library = face_library\ next to the script
python flask_insightface_face.py --port 5100         # specify the starting port
python flask_insightface_face.py --dir D:\faces      # specify the face library folder
python flask_insightface_face.py --library-info      # only count how many images and how many faces the face library has, then exit
```

The models are loaded first at startup (about 1 ~ 2 seconds), then the real address is printed:

```
Open http://127.0.0.1:5000 in your browser
```

If the port is taken, it automatically moves on to the next one (up to 20 attempts), so **always go by the address printed in the terminal**.

There are two panels on the same page, both submitted with AJAX, without reloading the whole page:

**① Query image vs candidate images (1:N)**
Pick 1 query image plus several candidate images (hold Ctrl in the browser to multi-select). Only the candidates uploaded this time are compared; the local face library is **not read at all**.

**② Query image vs face library folder (1:N)**
Pick 1 query image plus fill in a folder path, and compare it one by one against **all** the images in that folder, recursing into subdirectories.
The folder has a three-level priority: **the web input box > `INSIGHTFACE_FACE_LIBRARY_DIR` > the `face_library\` next to the script**,
so switching libraries just means pasting a path into the page—no environment variable to change, no restart.

Face library rules:

- Recursive scan, recognising `.jpg .jpeg .png .bmp .webp` (case-insensitive)
- When one image holds several faces, the first uses the filename, the rest are recorded as `filename #2`, `filename #3`
- Encoding results are cached by "file path + modification time": files that have not changed are not recomputed,
  and files that were modified or deleted are invalidated automatically —— **editing an image in the library needs no restart**
- Upload size limit: 64 MB per request; over that returns HTTP 413

The interface shows how many images and how many faces the current face library has, and ② re-counts automatically when the folder input box loses focus.

---

## ⌨Way 2: command line

```bash
# 1:1, compare two images once
python example2_insightface.py a.jpg b.jpg

# 1:N, one query image compared against every candidate in turn, ranked by descending similarity
python example2_insightface.py who-is-this.jpg user1.jpg user2.jpg handsome3.jpg

# 1:N, candidates can be given directly as a folder (collected recursively)
python example2_insightface.py who-is-this.jpg face_library -n 5

# Output JSON, convenient for scripts to process
python example2_insightface.py who-is-this.jpg face_library --json

# Double-click this file (run it with no arguments) to enter interactive mode
python example2_insightface.py
```

| Argument | Effect |
|---|---|
| `-n, --top N` | Show at most the top N entries; `0` or leaving it out means all |
| `--largest-only` | Compare only the largest single face in each image, instead of arbitrary pairs |
| `--no-recursive` | Only look at the first level of the candidate folder |
| `--no-bar` | Do not draw the similarity bar |
| `--json` | Output JSON |
| `-h, --help, info` | Print help |

Interactive mode menu: `1)` 1:1, `2)` 1:N, `0)` quit. Paths can be pasted directly;
quoted paths, relative paths and `%variable%` are all recognised; when a relative path cannot be found, the script's own directory is searched once more automatically.
Press Enter after reading the results to get back to the main menu, so you can run many rounds back to back.

Output details: the progress bar is a fixed 10 cells; with several candidates the similarity bars are scaled to **the lowest to the highest score of this run**
(ArcFace scores are crowded into the 40~70 range anyway, so drawing every one of them on a fixed 0~100 scale would make them all the same length),
and the actual scaling interval is printed below the table; **the number on the left is always the real similarity**;
with a single result, or when all results differ by less than 0.5, the real 0~100 scale is used instead.
The CLI lays out CJK text at a two-character width and keeps the table within 78 columns without wrapping;
symbols the terminal encoding cannot handle are downgraded automatically (`█` `░` `─` `…` → `#` `.` `-` `...`).

Exit codes: `0` completed normally; `1` no face was detected in one of the images, or an error occurred; `2` usage error / path does not exist / no usable candidates.

---

## 📚Way 3: use as a library

```python
from insightface_face_similarity import face_similarity_percent

percent = face_similarity_percent("a.jpg", "b.jpg")
if percent is None:          # no face detected in one of the images
    print("未检测到人脸")
else:
    print(f"{percent:.2f}%")
```

The standard way to write 1:N: the query image is computed once, each candidate is computed once, and after that it is all matrix multiplication.

```python
from insightface_face_similarity import encode_faces, similarity_matrix_percent

query = encode_faces("query.jpg")          # [query_vec, ...]
labels, rows = [], []
for path in candidate_paths:
    faces = encode_faces(path)
    if not faces:
        continue
    labels.append(path)
    rows.append(faces[0])                 # keep only the largest face in this image

scores = similarity_matrix_percent(query, rows)
ranking = sorted(zip(labels, scores), key=lambda r: r[1], reverse=True)
```

Main public interface:

| Function | Effect |
|---|---|
| `encode_faces(image)` | Returns one 512-dimensional float32 vector per face (already L2-normalized), **sorted by detection box area from largest to smallest**, so `[0]` is the largest face; returns `[]` if there is no face |
| `encode_faces_with_scores(image)` | Same, with each item additionally carrying `det_score` (detection confidence) and `area` |
| `face_similarity_percent(a, b, array_is_bgr=True, compare_all_faces=False)` | Similarity percentage of two images; returns `None` if there is no face |
| `cosine_similarity_percent(a, b)` | Compare two already-extracted vectors directly |
| `best_pair_percent(a, b, largest_only=False)` | The highest-scoring pair between two sets of vectors |
| `similarity_matrix_percent(query, candidates)` | The path taken by 1:N search; returns a 1-dimensional array of percentages |
| `to_bgr_array(image, array_is_bgr=True)` | Unify path / bytes / array into a BGR uint8 array |
| `get_app(det_threshold=None, det_size=None)` | Get the cached `FaceAnalysis`; the models are only really loaded on the first call, thread-safe |
| `warm_up()` | Warm up and return the elapsed time in milliseconds; call it once when the server starts |
| `find_model_directory()` | Look for `buffalo_l` in the lookup order; returns `None` if not found (= a download is needed) |
| `describe_configuration()` | A one-line configuration summary, for tracking down environment problems |

`image` accepts: file path / `pathlib.Path` / image bytes / numpy array
(the array can be BGR, RGB or grayscale; a float array whose maximum value does not exceed 1.0 is multiplied by 255 automatically).

The complete list of parameters, return structures and exceptions is in `使用说明_insightface.txt`.

---

## 🔌HTTP API

Everything returns JSON, so the web UI can call it directly as a backend.

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | the web page itself |
| `GET` | `/api/library?dir=...` | face library status: `{ok, path, images, faces}`; returns 400 if the folder does not exist |
| `POST` | `/api/query-set` | Panel ①. Form fields: `query` (single file), `candidates` (the same field name may be repeated), `top_results` |
| `POST` | `/api/query-library` | Panel ②. Form fields: `query`, `library_dir` (empty means the default), `top_results` |

`POST /api/query-set` returns on success:

```json
{
  "ok": true,
  "mode": "query-set",
  "query": "query.jpg",
  "query_faces": 1,
  "count": 5,
  "compared": 4,
  "matches": [
    { "rank": 1, "kind": "候选图片", "candidate": "user2.jpg",
      "similarity": 63.21, "bar_low": 41.5, "bar_high": 63.21, "best": true }
  ],
  "skipped": [ { "name": "bad-photo.jpg", "status": "未检测到人脸" } ],
  "elapsed_ms": 812
}
```

There are only two possible reasons in `skipped`: 「未检测到人脸」and 「处理失败：…」.
**One broken candidate does not affect the ranking of the other candidates.**

`top_results` is treated as 10 when it is omitted, an empty string or non-numeric; when it can be converted to an integer it is clamped to 1~100
(0 and negative numbers become 1, anything over 100 is truncated to 100).

`POST /api/query-library` returns `{ok, mode, query, query_faces, library, library_images, library_faces, matches, elapsed_ms}` on success.

Failures uniformly return HTTP 400 + `{"ok": false, "error": "<error message>"}`,
for example 「请选择一张查询图片」「人脸库目录不存在: …」「人脸库里没有检测到人脸: …」.
Uploads over 64 MB return HTTP 413.

curl example:

```bash
curl -F "query=@query.jpg" \
     -F "candidates=@1.jpg" -F "candidates=@2.jpg" \
     -F "top_results=5" \
     http://127.0.0.1:5000/api/query-set
```

---

## 🔧Environment variables

| Variable | Effect | Default |
|---|---|---|
| `INSIGHTFACE_MODEL_ROOT` | Root directory of the models | `~/.insightface` |
| `INSIGHTFACE_MODEL_DIR` | Directly specify the directory that holds the `.onnx` files, skipping the automatic lookup | empty |
| `INSIGHTFACE_PROVIDER` | onnxruntime execution backend, e.g. `CPUExecutionProvider` | automatic (CUDA is used when CUDA is detected) |
| `INSIGHTFACE_FACE_LIBRARY_DIR` | Face library folder for the web UI | the `face_library\` next to the script |
| `PORT` | Starting port for the web UI | `5000` |

### 📂 Model lookup order

Looked up in order; the first hit is used directly, and **the whole process never touches the network**:

1. The directory `INSIGHTFACE_MODEL_DIR` points at (or its parent plus `buffalo_l`)
2. `insightface_models\buffalo_l\` and `models\buffalo_l\` next to the script
3. `buffalo_l\` next to the script
4. Every subdirectory under the script's directory (searching for `buffalo_l` recursively)
5. `INSIGHTFACE_MODEL_ROOT\models\buffalo_l\`
6. `~/.insightface\models\buffalo_l\`

Only if none of those 6 locations is found does it fall back to InsightFace's official download flow.

**The least hassle for offline deployment**: copy the whole `buffalo_l` folder to the same
directory as `insightface_face_similarity.py`:

```
insightface_face_similarity.py
     buffalo_l\det_10g.onnx
     buffalo_l\w600k_r50.onnx
```

(Putting just those 2 files there is enough; the other 3 are not loaded.)
Once it is in place it **runs without any network access**. You can also point it somewhere else:

```bash
# Windows
set INSIGHTFACE_MODEL_DIR=D:\models\buffalo_l
# Linux / macOS
export INSIGHTFACE_MODEL_DIR=/opt/models/buffalo_l
```

---

## 📊Measured performance

Local environment: CPU only (no CUDA), Python 3.14, onnxruntime 1.30, insightface 2.0.

| Item | Value |
|---|---|
| Model loading (only 2 models loaded) | 1.1 ~ 1.9 s (cold versus warm disk cache makes a difference) |
| A single 2400x1080 image | 0.13 ~ 0.23 s |
| First search over a 500-image face library | about 1 minute (0.1 ~ 0.3 s per image) |
| Every search after that | matrix multiplication, tens of milliseconds |
| 1:1 cold start (including Python startup + model loading) | about 2.7 s |
| 1:1 with the models already loaded | about 0.6 ~ 0.7 s (about 0.3 s per image) |

The first search on a large library is slow because every image has to go through feature extraction once. After that the cache is used, and changing an image automatically recomputes just that one.

---

## 🐳Deployment

### 1. Simplest deployment (local or single machine)

```bash
python flask_insightface_face.py --port 5000 --dir D:\faces
```

> ⚠️ **The web UI listens on `127.0.0.1` only by default**, which means only this machine can reach it.
> This is deliberate: it is a service **with no login and no authentication**, and it reads images from
> whatever folder you point it at. **Do not** expose it directly to the public internet.
>
> If colleagues need it on the intranet, there are two ways to do it:
>
> - **Recommended**: reverse proxy with Nginx / Caddy, add HTTPS and access control, and leave Flask listening only on 127.0.0.1;
> - or change `app.run(host="127.0.0.1", ...)` at `flask_insightface_face.py:908`
>   to `host="0.0.0.0"`, **while at the same time** making sure there are firewall rules and an authentication layer in front of it.
>
> Under no circumstances should you point `/api/query-library` at a directory containing sensitive photos and then expose it,
> because this endpoint can read images from any directory on the server on demand.

### 2. GPU deployment

If there is an N card, swap onnxruntime for the GPU build:

```bash
python -m pip uninstall -y onnxruntime
python -m pip install onnxruntime-gpu
```

- **Never install both versions at the same time**;
- installing / upgrading insightface may put the CPU build of `onnxruntime` back; on an N-card machine you have to swap it again;
- `onnxruntime-gpu` has to match the local CUDA / cuDNN version, otherwise it falls back to CPU.

Confirm whether the GPU is really being used:

```bash
python insightface_face_similarity.py info
```

Only `providers=CUDAExecutionProvider,...` counts as success;
if it still shows `CPUExecutionProvider`, the GPU build is not installed or the version does not match.
This tool only enables the GPU automatically when CUDA is detected;
on macOS it will not pick CoreML automatically—set `INSIGHTFACE_PROVIDER` manually if you need it.

```bash
# force CPU only
export INSIGHTFACE_PROVIDER=CPUExecutionProvider
```

### 3. Offline or intranet deployment

1. Run `python insightface_face_similarity.py info` once on a machine with network access so it downloads the models;
2. Copy the `buffalo_l` folder (at least `det_10g.onnx` + `w600k_r50.onnx`) to the intranet machine;
3. Put it next to the script, or set `INSIGHTFACE_MODEL_DIR` to point at it;
4. Run `python insightface_face_similarity.py info` and confirm that `models=` points at a local path.

The Python dependencies themselves also have to be prepared offline in advance:

```bash
# machine with network access (on a GPU machine swap onnxruntime for onnxruntime-gpu)
python -m pip download insightface onnxruntime opencv-python numpy flask -d wheels

# intranet machine
python -m pip install --no-index --find-links=wheels insightface onnxruntime opencv-python numpy flask
```

### 4. Run as a long-running service

`app.run()` is Flask's built-in development server, suitable for this machine and small-scale intranet use.
For running it long term, or for more users, put a production-grade WSGI server in front:
Gunicorn on Linux, waitress on Windows. Neither can be started directly in the
`flask_insightface_face:app` form —— that would skip the model warm-up inside `main()`,
and the first request would have to absorb the 1~2 second load itself.

**This repository ships no built-in `wsgi.py`**; first create one in the repository root (it warms the models up before serving):

```python
# wsgi.py
import flask_insightface_face as web
from insightface_face_similarity import warm_up

warm_up()                 # load the models as soon as the process starts
application = web.app
```

```bash
python -m pip install waitress
waitress-serve --listen=127.0.0.1:5000 wsgi:application
```

On Linux the same applies with Gunicorn:

```bash
gunicorn --bind 127.0.0.1:5000 --workers 1 --threads 8 wsgi:application
```

> `--workers` is best left at 1: every worker loads its own copy of the models (about 300 MB of memory each),
> and the face library feature cache is **not shared across processes**. To squeeze the CPU, raising `--threads` is a better deal than adding workers.

systemd example:

```ini
# /etc/systemd/system/face-search.service
[Unit]
Description=InsightFace face similarity web
After=network.target

[Service]
WorkingDirectory=/opt/insightface-face-similarity
Environment=INSIGHTFACE_MODEL_DIR=/opt/models/buffalo_l
Environment=INSIGHTFACE_FACE_LIBRARY_DIR=/data/faces
ExecStart=/usr/local/bin/waitress-serve --listen=127.0.0.1:5000 wsgi:application
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now face-search
```

On Windows, use "Task Scheduler" to create a task that runs "at computer startup":

- Program: `python`
- Arguments: `-m waitress --listen=127.0.0.1:5000 wsgi:application`
- Start in: the repository directory

Using `python -m waitress` instead of `waitress-serve` saves you from having to guess which Python it was installed into.

### 5. Container deployment (not verified on this machine)

There is no Docker on this machine, so the `Dockerfile` below **has never actually been run**,
so please verify it yourself before first use; the paths tested and working in this repository are sections 1~4 above.

```dockerfile
FROM python:3.11-slim

# opencv-python needs these two system libraries, they are not present in the slim image by default
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir insightface onnxruntime opencv-python numpy flask waitress

# this repository ships no wsgi.py, create one in the container now and warm the models up
RUN printf 'import flask_insightface_face as web\nfrom insightface_face_similarity import warm_up\nwarm_up()\napplication = web.app\n' > /app/wsgi.py

# the models are not in the repo; on first container start they download automatically into ~/.insightface (network access required).
# in an offline environment change this to volume-mount an already-downloaded buffalo_l and set INSIGHTFACE_MODEL_DIR.
VOLUME /root/.insightface

EXPOSE 5000

# a WSGI server must listen on 0.0.0.0:
# running flask_insightface_face.py directly only listens on 127.0.0.1, which cannot be reached from outside the container.
CMD ["waitress-serve", "--listen=0.0.0.0:5000", "wsgi:application"]
```

```bash
docker build -t face-search .
# only map to the loopback address of this machine; only consider -p 5000:5000 after confirming it is safe
docker run --rm -p 127.0.0.1:5000:5000 -v /data/faces:/data/faces face-search
```

---

## 🆘FAQ

**Q Reports `No face was detected in one of the images`**
No face was detected in one of the images. Extreme profile views, small faces and blur can all cause this.
You can lower the detection threshold (`get_app(det_threshold=0.3)`), or crop the image to around the face first and try again.

**Q It hangs on the first run, then reports a network error**
It is downloading the `buffalo_l` model bundle (about 326 MB), which needs network access.
Once it is downloaded, put it in `~/.insightface/models/buffalo_l/`, and after that no network is needed.
You can also download it by hand (the official Model Zoo table has a Download link for `buffalo_l`),
unpack the `buffalo_l` folder and put it next to the script.

**Q How do I confirm the models are installed correctly**
`python insightface_face_similarity.py info` prints the model name, the dimension, the execution backend and the model location.

**Q Still slow after installing the CUDA build of onnxruntime**
Run `python insightface_face_similarity.py info` and check whether `providers` is
`CUDAExecutionProvider`. A displayed `CPUExecutionProvider` means `onnxruntime-gpu`
is not installed, or was swapped back to the CPU build when insightface was installed, or does not match the CUDA version.

**Q The web page opens on 5001 instead of 5000**
5000 is taken by another program, so the app moves on automatically; **always go by the address printed in the terminal**. You can also set it with `--port`.

**Q Do I have to restart after changing files in the face library**
No. The library is cached by "file path + modification time" and anything that changes is recomputed automatically;
changing the path in the folder input box does not need a restart either.

**Q The first search over the face library is very slow**
The first time it has to run feature extraction over the whole directory, 0.1 ~ 0.3 seconds per image, so 500 images take about 1 minute.
After that everything goes through the cache, tens of milliseconds. The terminal prints progress like 「载入人脸库 12/500」.

**Q The window flashes and disappears after a double-click**
Running it with no arguments stops at the main menu and waits for your input.
If it really does flash and exit, the Python file association is broken; run it from the command line to see the error.

**Q All the scores are very close, so I cannot tell who is who**
The similarity bars and the command line table are already scaled to this run's score range; **the number on the left is the real score**,
so just read the numbers.

**Q The same face scores 70% on its own but only 50% when searched inside a large group photo**
That is normal. The detector looks at the whole image at 640x640, so the bigger the image the fewer pixels each face gets, and the features are not that stable.
Crop the image to around the face before comparing and the score comes back to 70%.

**Q Why can two different people also reach 50%**
That is just how ArcFace is; look-alikes score very high. So this tool **does not decide for you** whether two photos are the same person,
and the threshold has to be calibrated with your own data.

**Q Does the web UI have a mobile layout**
Yes, the page uses Bootstrap 5, works fine in mobile browsers, and desktop and mobile get the same interface.

**Q Will the photos be uploaded to the internet**
No. Decoding, detection, feature extraction and comparison are all done inside the local process,
and the web UI is local Flask processing the uploaded files directly. insightface sets `ORT_DISABLE_TELEMETRY=1`
before it imports onnxruntime.

---

## 🙏Acknowledgements and citations

This project **implements no face algorithm of its own**; all of the algorithmic capability comes from the official open-source InsightFace project.
Without it this repository would not exist—many thanks to the maintainers, Jia Guo, Jiankang Deng and others.

- **Official InsightFace repository**: <https://github.com/deepinsight/insightface>
- Official website: <https://insightface.ai>
- PyPI: <https://pypi.org/project/insightface/>
- Model Zoo (model bundles, download links, official accuracies): <https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md>
- Notes on the runtime environment and execution backends: <https://github.com/deepinsight/insightface/blob/master/python-package/docs/runtime.md>
- Related papers:
  - SCRFD (this tool's detector) —
    Guo, Deng, Lattas, Zafeiriou. *Sample and Computation Redistribution for
    Efficient Face Detection.* [arXiv:2105.04714](https://arxiv.org/abs/2105.04714)
  - ArcFace (this tool's feature extractor) —
    Deng, Guo, Yang, Xue, Kotsia, Zafeiriou. *ArcFace: Additive Angular Margin
    Loss for Deep Face Recognition.* [arXiv:1801.07698](https://arxiv.org/abs/1801.07698)

Also thanks to [ONNX Runtime](https://onnxruntime.ai) (the inference backend) and
[Flask](https://flask.palletsprojects.com/) (the web UI).

Issues, pull requests, translations, or your own threshold calibration experience are all welcome—post them and let's discuss them together.

---

## 📜License

- **Code in this repository: MIT License**, which permits commercial use, modification and closed-source distribution, as long as the copyright notice is kept.
  Please add a `LICENSE` file when you publish.
  If you need explicit patent grant terms, switch to Apache-2.0 (which also permits commercial use; both are more permissive than GPL).
- **The model weights are not in this repository**: the InsightFace code is MIT, but the official pretrained weights
  are **for non-commercial research use only**. For commercial use, contact the official licensor, or switch to a model you have the right to use.
  See [Model licenses](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md#model-licenses).

> Disclaimer: this tool **does not perform face identification**; the score is only a cosine similarity,
> and both the threshold and its consequences are for the user to judge; when processing other people's faces, obey local privacy regulations and obtain their consent.
