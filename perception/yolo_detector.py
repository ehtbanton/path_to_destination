from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class Detection:
    cls: str
    conf: float
    xyxy: Tuple[int, int, int, int]

class YOLODetector:
    def __init__(self, model_path: str = "yolov8n.pt", img_size: int = 640):
        self.img_size = img_size
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise RuntimeError("Install ultralytics: pip install ultralytics") from e

        self.model = YOLO(model_path)

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        results = self.model.predict(source=bgr, imgsz=self.img_size, verbose=False)
        r0 = results[0]

        names = r0.names
        dets: List[Detection] = []
        if r0.boxes is None:
            return dets

        for b in r0.boxes:
            cls_id = int(b.cls.item())
            conf = float(b.conf.item())
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            dets.append(Detection(cls=names[cls_id], conf=conf, xyxy=(x1, y1, x2, y2)))
        return dets
