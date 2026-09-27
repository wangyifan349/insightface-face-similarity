"""
InsightFace ArcFace (buffalo_l) face similarity, returned as a cosine
similarity percentage. Runs through onnxruntime, so no GPU is required.

This is the InsightFace counterpart of dlib_face_similarity.py: same public
interface, same "how similar do these two look" answer, different engine and a
different number scale.

    dlib       128-d descriptor, same person usually 85%+ on this scale
    ArcFace    512-d descriptor, same person usually 45-70% on this scale

The number is a plain cosine similarity of the two 512-d vectors, so it is NOT
comparable with the dlib percentage. It also does not answer "is this the same
person" - pick your own threshold from your own samples.

The buffalo_l pack is used and nothing else. It pairs an SCRFD det_10g
detector with an ArcFace w600k_r50 recogniser, which is the most accurate of
the InsightFace packs and still runs at roughly 0.1-0.3 s per photo on a CPU.

    buffalo_l/det_10g.onnx     face detection
    buffalo_l/w600k_r50.onnx   512-d face embedding

Only those two models are loaded. The landmark (1k3d68, 143 MB) and gender/age
models that ship in the same folder are skipped because a similarity score does
not need them, which roughly halves load time and memory.

The pack is downloaded once into ~/.insightface/models; an internet connection
is needed only for that first run.

Usage:

    from insightface_face_similarity import face_similarity_percent

    percent = face_similarity_percent("a.jpg", "b.jpg")
    if percent is not None:
        print(f"{percent:.2f}%")
"""

from __future__ import annotations

import os
import threading
import warnings
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np

ImageInput = Union[str, Path, bytes, bytearray, np.ndarray]

MODEL_NAME = "buffalo_l"

# The two .onnx files this tool actually needs out of the pack.
DETECTOR_FILE = "det_10g.onnx"
RECOGNIZER_FILE = "w600k_r50.onnx"
REQUIRED_FILES = (DETECTOR_FILE, RECOGNIZER_FILE)

MODEL_ROOT_ENVIRONMENT_VARIABLE = "INSIGHTFACE_MODEL_ROOT"
MODEL_DIRECTORY_ENVIRONMENT_VARIABLE = "INSIGHTFACE_MODEL_DIR"
PROVIDER_ENVIRONMENT_VARIABLE = "INSIGHTFACE_PROVIDER"

DESCRIPTOR_DIMENSION = 512
DET_SIZE = (640, 640)
DET_THRESHOLD = 0.5

# InsightFace's landmark_3d_68 model calls a deprecated scikit-image API and
# warns on every single face. The warning is noise here, so it is muted once.
warnings.filterwarnings("ignore", message=r".*estimate.*is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface.*")

# FaceAnalysis instances, so nothing is ever built twice.
_APPS: dict = {}
_APPS_LOCK = threading.Lock()

# FaceAnalysis keeps no record of its own providers, so it is tracked here.
_ACTIVE_PROVIDERS: list = []


# -------------------- Model Discovery --------------------
def model_root() -> Path:
    """Base folder the buffalo_l pack lives in."""
    configured = os.environ.get(MODEL_ROOT_ENVIRONMENT_VARIABLE)
    return Path(configured if configured else "~/.insightface").expanduser()


def default_model_directory() -> Path:
    """Where the pack ends up if it has to be downloaded."""
    return model_root() / "models" / MODEL_NAME


def _holds_pack(directory: Path) -> bool:
    return all((directory / file_name).is_file() for file_name in REQUIRED_FILES)


def find_model_directory() -> Optional[Path]:
    """Locate a folder that already holds the buffalo_l .onnx files.

    Looks, in order, at:
      1. INSIGHTFACE_MODEL_DIR, and INSIGHTFACE_MODEL_DIR/buffalo_l
      2. insightface_models/buffalo_l and models/buffalo_l next to this file
      3. buffalo_l next to this file, and any matching sub-directory of it
      4. INSIGHTFACE_MODEL_ROOT/models/buffalo_l
      5. ~/.insightface/models/buffalo_l

    Returns None when nothing suitable is on disk, which lets FaceAnalysis fall
    back to downloading the pack.
    """
    script_directory = Path(__file__).resolve().parent

    candidates = []

    configured = os.environ.get(MODEL_DIRECTORY_ENVIRONMENT_VARIABLE)
    if configured:
        base = Path(configured).expanduser()
        candidates.extend([base / MODEL_NAME, base])

    candidates.extend(
        [
            script_directory / "insightface_models" / MODEL_NAME,
            script_directory / "models" / MODEL_NAME,
            script_directory / MODEL_NAME,
            default_model_directory(),
        ]
    )

    if script_directory.is_dir():
        candidates.extend(sorted(script_directory.rglob(MODEL_NAME)))

    for candidate in candidates:
        if candidate.is_dir() and _holds_pack(candidate):
            return candidate
    return None


# -------------------- Execution Provider --------------------
def resolve_providers() -> list:
    """Execution providers for onnxruntime, CUDA first when it is usable.

    INSIGHTFACE_PROVIDER overrides the choice, for example
    INSIGHTFACE_PROVIDER=CPUExecutionProvider.
    """
    forced = os.environ.get(PROVIDER_ENVIRONMENT_VARIABLE)
    if forced:
        return [part.strip() for part in forced.split(",") if part.strip()]

    try:
        import onnxruntime

        available = onnxruntime.get_available_providers()
    except Exception:
        available = []

    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def active_providers() -> list:
    """Providers the loaded app was actually built with."""
    with _APPS_LOCK:
        return list(_ACTIVE_PROVIDERS)


# -------------------- Model Loading --------------------
def get_app(det_threshold: Optional[float] = None, det_size: Optional[Sequence[int]] = None):
    """Return the cached FaceAnalysis app, building it on first call.

    Args:
        det_threshold: face detection confidence, default 0.5.
        det_size: detection input size, default (640, 640).
    """
    threshold = DET_THRESHOLD if det_threshold is None else float(det_threshold)
    size = tuple(DET_SIZE if det_size is None else det_size)

    key = (threshold, size, tuple(resolve_providers()))
    cached = _APPS.get(key)
    if cached is not None:
        return cached

    with _APPS_LOCK:
        cached = _APPS.get(key)
        if cached is not None:
            return cached

        from insightface.app import FaceAnalysis

        providers = list(key[2])
        # ctx_id only means something for the CUDA provider.
        ctx_id = 0 if providers[0] == "CUDAExecutionProvider" else -1

        local_directory = find_model_directory()
        if local_directory is not None:
            # Already on disk: load it in place, no download, no network.
            app = FaceAnalysis(
                name=str(local_directory),
                providers=providers,
                allowed_modules=["detection", "recognition"],
            )
        else:
            app = FaceAnalysis(
                name=MODEL_NAME,
                root=str(model_root()),
                providers=providers,
                allowed_modules=["detection", "recognition"],
            )

        app.prepare(ctx_id=ctx_id, det_size=list(size), det_thresh=threshold)
        _APPS[key] = app
        if not _ACTIVE_PROVIDERS:
            _ACTIVE_PROVIDERS.extend(providers)
        return app


def warm_up() -> float:
    """Load the model and return how many milliseconds it took.

    The first call in a process always pays the full model load, so a server
    can call this at startup to keep it off the first user's request.
    """
    import time

    started = time.perf_counter()
    get_app()
    return (time.perf_counter() - started) * 1000.0


# -------------------- Image Input --------------------
def to_bgr_array(image: ImageInput, array_is_bgr: bool = True) -> np.ndarray:
    """Normalize any supported input into a contiguous BGR uint8 array."""
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            array = np.repeat(image[:, :, None], 3, axis=2)
        else:
            array = image if array_is_bgr else image[:, :, ::-1]
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating) and array.size:
                if float(np.nanmax(array)) <= 1.0:
                    array = array * 255.0
            array = np.clip(array, 0, 255).astype("uint8")
        return np.ascontiguousarray(array)

    if isinstance(image, (bytes, bytearray)):
        encoded_bytes = bytes(image)
    else:
        image_path = Path(image)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image does not exist: {image_path}")
        # cv2.imread cannot open non-ASCII paths on Windows, so read the bytes.
        encoded_bytes = image_path.read_bytes()

    decoded = cv2.imdecode(np.frombuffer(encoded_bytes, dtype="uint8"), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Unsupported or corrupted image data")
    return np.ascontiguousarray(decoded)


# -------------------- Encoding --------------------
def _face_area(face) -> float:
    x1, y1, x2, y2 = (float(value) for value in face.bbox[:4])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _normalized(vector) -> Optional[np.ndarray]:
    array = np.asarray(vector, dtype="float32").reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return None
    return array / norm


def encode_faces(image: ImageInput, array_is_bgr: bool = True) -> list:
    """Return one L2 normalized 512-d ArcFace embedding per detected face.

    Sorted by detection box area from large to small, so element 0 is the
    biggest face in the image, exactly like the dlib version of this function.
    Returns an empty list when no face is found.
    """
    bgr_array = to_bgr_array(image, array_is_bgr)
    faces = get_app().get(bgr_array)
    if not faces:
        return []

    embeddings = []
    for face in faces:
        raw = getattr(face, "embedding", None)
        if raw is None:
            continue
        vector = _normalized(raw)
        if vector is not None:
            embeddings.append((_face_area(face), vector))

    embeddings.sort(key=lambda row: row[0], reverse=True)
    return [vector for _area, vector in embeddings]


def encode_faces_with_scores(
    image: ImageInput, array_is_bgr: bool = True
) -> list:
    """Like encode_faces, but each item is {"embedding", "det_score", "area"}.

    Useful when you want to show how sure the detector was next to the score.
    """
    bgr_array = to_bgr_array(image, array_is_bgr)
    faces = get_app().get(bgr_array)
    results = []
    for face in faces:
        raw = getattr(face, "embedding", None)
        if raw is None:
            continue
        vector = _normalized(raw)
        if vector is None:
            continue
        results.append(
            {
                "embedding": vector,
                "det_score": float(getattr(face, "det_score", 0.0)),
                "area": _face_area(face),
            }
        )
    results.sort(key=lambda row: row["area"], reverse=True)
    return results


# -------------------- Similarity --------------------
def cosine_similarity_percent(
    embedding_a: np.ndarray, embedding_b: np.ndarray
) -> float:
    """Cosine similarity of two embeddings, expressed as a percentage."""
    first = np.asarray(embedding_a, dtype="float32").reshape(-1)
    second = np.asarray(embedding_b, dtype="float32").reshape(-1)
    if first.size != second.size:
        raise ValueError(f"Embedding sizes differ: {first.size} and {second.size}")
    return float(np.dot(first, second)) * 100.0


def best_pair_percent(
    embeddings_a: Sequence, embeddings_b: Sequence, largest_only: bool = False
) -> float:
    """Highest similarity between any face of A and any face of B.

    Group photos work out of the box. Pass largest_only=True to compare only
    the biggest face of each side instead.
    """
    if largest_only:
        return cosine_similarity_percent(embeddings_a[0], embeddings_b[0])
    return max(
        cosine_similarity_percent(first, second)
        for first in embeddings_a
        for second in embeddings_b
    )


def similarity_matrix_percent(
    query_embeddings: Sequence, candidate_embeddings: Sequence
) -> np.ndarray:
    """Best score for every candidate, as a 1-D array of percentages.

    One matrix multiply instead of a Python loop, which is what makes 1:N over a
    large folder fast once the candidates are encoded.
    """
    query_matrix = np.vstack(query_embeddings).astype("float32")
    candidate_matrix = np.vstack(candidate_embeddings).astype("float32")
    return (candidate_matrix @ query_matrix.T).max(axis=1) * 100.0


def face_similarity_percent(
    image_a: ImageInput,
    image_b: ImageInput,
    array_is_bgr: bool = True,
    compare_all_faces: bool = False,
) -> Optional[float]:
    """Similarity between the faces in two images, as a percentage.

    Args:
        image_a: image path, encoded image bytes, or a NumPy array.
        image_b: image path, encoded image bytes, or a NumPy array.
        array_is_bgr: set to False when a NumPy array is already in BGR order.
        compare_all_faces: use the best matching face pair instead of the
            largest face of each image.

    Returns:
        A float percentage, or None when no face is found in one of the images.
        Higher means more similar. The value is a cosine similarity in
        [-100, 100]; on this scale two photos of the same person usually land
        around 45-70 and two different people around 0-40.
    """
    embeddings_a = encode_faces(image_a, array_is_bgr)
    embeddings_b = encode_faces(image_b, array_is_bgr)

    if not embeddings_a or not embeddings_b:
        return None

    if embeddings_a[0].size != DESCRIPTOR_DIMENSION:
        raise ValueError(f"Unexpected embedding size: {embeddings_a[0].size}")

    similarity = best_pair_percent(embeddings_a, embeddings_b, compare_all_faces)

    # Guard against floating point drift outside the valid range.
    return float(min(100.0, max(-100.0, similarity)))


def describe_configuration() -> str:
    """One line summary of model, provider and score scale, for diagnostics."""
    providers = ", ".join(active_providers()) or "not loaded yet"
    directory = find_model_directory()
    location = str(directory) if directory else f"{default_model_directory()} (downloads on first run)"
    return (
        f"model={MODEL_NAME} dim={DESCRIPTOR_DIMENSION} "
        f"providers={providers} det={DET_SIZE[0]}x{DET_SIZE[1]}@{DET_THRESHOLD} "
        f"models={location}"
    )


if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) == 2 and _sys.argv[1] in ("-h", "--help", "info"):
        print("Usage: python insightface_face_similarity.py <image_a> <image_b>")
        print(describe_configuration())
        raise SystemExit(0)

    if len(_sys.argv) != 3:
        print("Usage: python insightface_face_similarity.py <image_a> <image_b>")
        raise SystemExit(1)

    try:
        score = face_similarity_percent(_sys.argv[1], _sys.argv[2])
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Failed: {error}")
        raise SystemExit(1)

    if score is None:
        print("No face was detected in one of the images")
    else:
        print(f"Similarity: {score:.2f}%")
