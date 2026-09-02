#!/usr/bin/env python3
"""Download the dlib 68-point predictor needed by SourceCode.py (live EAR demo)."""

from __future__ import annotations

import bz2
import os
import sys
import urllib.request

PREDICTOR = "shape_predictor_68_face_landmarks.dat"
ARCHIVE = PREDICTOR + ".bz2"
URL = "http://dlib.net/files/" + ARCHIVE


def main() -> int:
    root = os.path.dirname(os.path.abspath(__file__))
    predictor_path = os.path.join(root, PREDICTOR)
    archive_path = os.path.join(root, ARCHIVE)

    if os.path.exists(predictor_path) and os.path.getsize(predictor_path) > 0:
        print(f"Already present: {predictor_path}")
        return 0

    print(f"Downloading {URL} …")
    try:
        urllib.request.urlretrieve(URL, archive_path)
    except Exception as e:
        print(f"Download failed: {e}")
        print(f"Manually download {URL}, decompress, and place {PREDICTOR} in the project root.")
        return 1

    print(f"Decompressing {ARCHIVE} …")
    try:
        with bz2.open(archive_path, "rb") as src, open(predictor_path, "wb") as dst:
            dst.write(src.read())
    except Exception as e:
        print(f"Decompress failed: {e}")
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
