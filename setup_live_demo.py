#!/usr/bin/env python3
"""Download the dlib 68-point predictor needed by SourceCode.py (live EAR demo)."""

from __future__ import annotations

import bz2
import hashlib
import os
import sys
import tempfile
import urllib.request

PREDICTOR = "shape_predictor_68_face_landmarks.dat"
ARCHIVE = PREDICTOR + ".bz2"
URL = "https://dlib.net/files/" + ARCHIVE
# Upstream file is ~95 MB; reject truncated leftovers from a failed setup.
MIN_PREDICTOR_BYTES = 50 * 1024 * 1024
# SHA-256 of the official decompressed shape_predictor_68_face_landmarks.dat
PREDICTOR_SHA256 = (
    "fbdc2cb80eb9aa7a758672cbfdda32ba6300efe9b6e6c7a299ff7e736b11b92f"
)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _predictor_ready(path: str) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) < MIN_PREDICTOR_BYTES:
        return False
    return _sha256_file(path) == PREDICTOR_SHA256


def main() -> int:
    root = os.path.dirname(os.path.abspath(__file__))
    predictor_path = os.path.join(root, PREDICTOR)
    archive_path = os.path.join(root, ARCHIVE)

    if _predictor_ready(predictor_path):
        print(f"Already present: {predictor_path}")
        return 0

    if os.path.exists(predictor_path):
        print(
            f"Removing incomplete/untrusted predictor "
            f"({os.path.getsize(predictor_path)} bytes)…"
        )
        os.remove(predictor_path)

    print(f"Downloading {URL} …")
    try:
        urllib.request.urlretrieve(URL, archive_path)
    except Exception as e:
        print(f"Download failed: {e}")
        print(f"Manually download {URL}, decompress, and place {PREDICTOR} in the project root.")
        return 1

    print(f"Decompressing {ARCHIVE} …")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=root, prefix=PREDICTOR + ".", suffix=".tmp", delete=False
        ) as tmp:
            tmp_path = tmp.name
            with bz2.open(archive_path, "rb") as src:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    tmp.write(chunk)
            tmp.flush()
            os.fsync(tmp.fileno())

        if os.path.getsize(tmp_path) < MIN_PREDICTOR_BYTES:
            raise RuntimeError(
                f"decompressed file too small ({os.path.getsize(tmp_path)} bytes)"
            )

        actual = _sha256_file(tmp_path)
        if actual != PREDICTOR_SHA256:
            raise RuntimeError(
                f"SHA-256 mismatch (got {actual}, expected {PREDICTOR_SHA256})"
            )

        os.replace(tmp_path, predictor_path)
        tmp_path = None
    except Exception as e:
        print(f"Decompress/verify failed: {e}")
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return 1
    finally:
        if os.path.exists(archive_path):
            os.remove(archive_path)

    size_mb = os.path.getsize(predictor_path) / (1024 * 1024)
    print(f"Ready: {predictor_path} ({size_mb:.1f} MB)")
    print("Next: python SourceCode.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
