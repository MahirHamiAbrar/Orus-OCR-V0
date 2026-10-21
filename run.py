"""Orus-OCR-0.1 — run OCR on images with the frozen release weights in ./models.

  python run.py page.jpg [more.jpg ...]          # language, handwritten Yes/No, structured text
  python run.py page.jpg --json                   # full JSON (lines with boxes and confidence, timings)
  python run.py --demo                            # self-check on a rendered mixed Bangla+English page

Python:
  import sys; sys.path.insert(0, "orus")
  from ocr import OCR
  r = OCR(release="models")("page.jpg")
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "orus"))

import ocr  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args == ["--demo"]:
        ocr.demo()                                   # uses ./models (orus_rec.onnx, det.onnx, LM, picture net, hand flag)
    else:
        if "--release" not in args:
            args += ["--release", os.path.join(HERE, "models")]
        sys.argv = [sys.argv[0]] + args
        ocr.main()
