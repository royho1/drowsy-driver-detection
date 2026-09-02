# Project state

Durable agent memory for this repository. Not a README. Capture only what future sessions would otherwise rediscover the hard way.

## Snapshot

- **What this is:** Python drowsy-driver prototype with two separate demos: real-time EAR webcam (`SourceCode.py`) and Flask image-upload classification (`app.py`).
- **Current phase / constraints:** `drowsiness_model.h5` is not in the repo and there is no training code here; `app.py` must not load it at import time.

## Learnings

### 2026-09-02 — Missing Keras model must not crash import
- **Learning:** `drowsiness_model.h5` was removed from the repo with no training script left behind. Eager `load_model` at import made `app.py` / `test_data.py` unusable.
- **Why it matters:** Future agents may try to restore training or re-add a binary model; the intended fix is lazy load + clear errors, not new TensorFlow training code unless explicitly requested.
- **Implication:** Keep model loading inside request/script entry points; UI and JSON should explain that the upload demo needs a local `.h5`.

## Nuances and gotchas

- Real project path is under `Desktop/Personal/Drowsy Driver Detection/...` (not a bare `Desktop/Drowsy Driver Detection` copy).
- `index.html` lives in the repo root; Flask is configured with `template_folder="."`.
- `SourceCode.py` (EAR + dlib) does not need `drowsiness_model.h5`; only the Flask upload demo does.

## Decisions

- Do not add TensorFlow training code unless the user asks for it.
- Prefer graceful degradation (503 JSON + banner) over inventing a placeholder model.

## Do not store

Secrets, tokens, `.env` values, session transcripts, or content that already lives in the README unless it is easy to miss.
