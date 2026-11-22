"""Korean OCR: PaddleOCR's PP-OCRv5 Korean recognizer, run with ONNX Runtime through RapidOCR."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from visionpilot_mcp.schemas import BBox, TextLine

OCR_MODEL_NAME = "PP-OCRv6 det + korean_PP-OCRv5_mobile_rec (RapidOCR, ONNX)"


def load_korean_ocr(cpu_threads: int) -> Any:
    from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

    # Detection uses the PP-OCRv6 detector that ships inside the rapidocr package (script-agnostic).
    # Recognition uses the Korean PP-OCRv5 model, downloaded once (~10 MB) into the rapidocr package
    # folder inside .venv, so it stays on the project drive.
    return RapidOCR(
        params={
            "Global.log_level": "warning",
            "Global.use_cls": False,
            "Rec.lang_type": LangRec.KOREAN,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.model_type": ModelType.MOBILE,
            "EngineConfig.onnxruntime.intra_op_num_threads": cpu_threads,
        }
    )


def run_ocr(engine: Any, image: Image.Image, *, min_confidence: float) -> list[TextLine]:
    output = engine(np.asarray(image)[:, :, ::-1].copy(), text_score=min_confidence)  # RapidOCR expects BGR
    if output.txts is None or output.boxes is None:
        return []

    lines: list[TextLine] = []
    for polygon, text, score in zip(output.boxes, output.txts, output.scores, strict=True):
        if not str(text).strip():
            continue
        points = [(round(float(x), 1), round(float(y), 1)) for x, y in polygon]
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        lines.append(
            TextLine(
                text=str(text).strip(),
                confidence=round(float(score), 3),
                box=BBox.from_xyxy(min(xs), min(ys), max(xs), max(ys)),
                polygon=points,
            )
        )
    if lines:
        # Reading order: top to bottom, then left to right. Lines whose centers fall in the same
        # band (about half a typical line height) count as one row.
        band = max(float(np.median([ln.box.y2 - ln.box.y1 for ln in lines])) * 0.5, 1.0)
        lines.sort(key=lambda ln: (int(((ln.box.y1 + ln.box.y2) / 2) // band), ln.box.x1))
    return lines
