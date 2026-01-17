from __future__ import annotations
import numpy as np

def robust_affine_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Fit y ≈ a*x + b using a robust-ish trick:
    - remove NaNs
    - trim extremes (percentile clip)
    - least squares on remaining
    """
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m].astype(np.float32)
    y = y[m].astype(np.float32)
    if x.size < 200:
        raise RuntimeError("Not enough overlap pixels to fit scale")

    # Trim outliers
    xl, xh = np.percentile(x, [5, 95])
    yl, yh = np.percentile(y, [5, 95])
    m2 = (x >= xl) & (x <= xh) & (y >= yl) & (y <= yh)
    x = x[m2]
    y = y[m2]

    A = np.stack([x, np.ones_like(x)], axis=1)
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)

def scale_midas_to_meters(midas_rel: np.ndarray, tof_m: np.ndarray) -> np.ndarray:
    """
    Convert MiDaS relative depth map into approximate meters using ToF overlap.

    We use inverse-depth mapping: tof ≈ a*(1/(midas+eps)) + b
    """
    eps = 1e-6
    inv = 1.0 / (midas_rel.astype(np.float32) + eps)

    # Fit using overlap where ToF is valid
    a, b = robust_affine_fit(inv, tof_m.astype(np.float32))

    meters = a * inv + b
    meters[~np.isfinite(meters)] = np.nan
    meters[meters <= 0] = np.nan
    return meters
