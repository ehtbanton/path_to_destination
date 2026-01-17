from __future__ import annotations
import numpy as np

class MiDaSDepth:
    def __init__(self, device: str = "cpu"):
        import torch  # type: ignore
        self.torch = torch
        self.device = device

        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.model.to(device).eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform

    def predict(self, bgr: np.ndarray) -> np.ndarray:
        """
        Returns relative depth (H, W) float32 (higher/ lower depends on model; we treat it as relative).
        """
        import cv2
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = self.transform(img).to(self.device)

        with self.torch.no_grad():
            pred = self.model(inp)
            pred = self.torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        d = pred.cpu().numpy().astype(np.float32)
        return d
