# Project state

Durable agent memory for this repository. Not a README. Capture only what future sessions would otherwise rediscover the hard way.

## Snapshot

- **What this is:** Python drowsy-driver prototype with two separate demos: real-time EAR webcam (`SourceCode.py`) and Flask image-upload classification (`app.py`).
- **Current phase / constraints:** `drowsiness_model.h5` and `shape_predictor_68_face_landmarks.dat` are not in the repo; no training code here. Prefer clear setup errors over inventing assets.

## Learnings

### 2026-09-02 — Clone setup mismatches
- **Learning:** README still referenced `drunk_drowsy_detection.py` / `data/` after the fork; live demo needs a separately downloaded dlib predictor; dataset scripts expect `kaggle_data/`.
- **Why it matters:** Forks look “broken” even when the EAR pipeline is fine.
- **Implication:** Keep README/structure/`requirements.txt` aligned with real files; fail with download instructions when predictor or `kaggle_data/` is missing.

### 2026-09-02 — Missing Keras model must not crash import
- **Learning:** `drowsiness_model.h5` was removed from the repo with no training script left behind. Eager `load_model` at import made `app.py` / `test_data.py` unusable.
- **Why it matters:** Future agents may try to restore training or re-add a binary model; the intended fix is lazy load + clear errors, not new TensorFlow training code unless explicitly requested.
- **Implication:** Keep model loading inside request/script entry points; UI and JSON should explain that the upload demo needs a local `.h5`.

## Nuances and gotchas

- Real project path is under `Desktop/Personal/Drowsy Driver Detection/...` (not a bare `Desktop/Drowsy Driver Detection` copy).
- `index.html` lives in the repo root; Flask is configured with `template_folder="."`.
- `SourceCode.py` (EAR + dlib) does not need `drowsiness_model.h5`; only the Flask upload demo does.
- Upstream fork note: README credits `varshathennarasu/DrowsyDriverDetection`.

## Decisions

- Do not add TensorFlow training code unless the user asks for it.
- Prefer graceful degradation (clear errors / banners) over inventing placeholder models or datasets.

## Do not store

Secrets, tokens, `.env` values, session transcripts, or content that already lives in the README unless it is easy to miss.
