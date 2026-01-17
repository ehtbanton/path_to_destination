from __future__ import annotations
import numpy as np

class ToFReader:
    """
    Reads depth in meters from an Arducam ToF module.

    Notes:
    - Most Arducam ToF examples expose depth as either mm or float meters depending on API.
    - We standardize output to a (H, W) float32 numpy array in meters, with np.nan for invalid pixels.
    """

    def __init__(self, range_mode_m: int = 4):
        self.range_mode_m = range_mode_m

        # Lazy import so your repo can run without the SDK installed.
        try:
            # This import name depends on the SDK you install.
            # You will likely replace this with the actual module from Arducam’s package.
            import arducam_tof  # type: ignore
            self._sdk = arducam_tof
        except Exception as e:
            raise RuntimeError(
                "Arducam ToF SDK not found. Install Arducam ToF SDK/Python bindings first."
            ) from e

        # PSEUDO-CODE: replace with actual SDK init calls
        self.cam = self._sdk.Camera()
        self.cam.open()
        self.cam.set_range(self.range_mode_m)  # often 2 or 4 meters
        self.cam.start()

    def read_depth_m(self) -> np.ndarray:
        """
        Returns depth in meters as float32 array (H, W). Invalid pixels -> np.nan
        """
        frame = self.cam.request_frame()  # replace with actual SDK call
        if frame is None:
            raise RuntimeError("No ToF frame received")

        depth = frame.get_depth()  # replace with actual SDK call
        depth = np.asarray(depth, dtype=np.float32)

        # If SDK returns millimeters, convert:
        # depth = depth / 1000.0

        # Example invalid handling (adjust depending on SDK):
        depth[(depth <= 0) | (depth > float(self.range_mode_m) + 0.25)] = np.nan
        return depth

    def close(self) -> None:
        try:
            self.cam.stop()
            self.cam.close()
        except Exception:
            pass
