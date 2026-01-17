from __future__ import annotations
import time
import numpy as np

def main():
    import cv2

    # ---- Camera input (placeholder) ----
    # Replace this with PiCamera2 capture on device.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")
    
    from perception.tof_arducam import ToFReader
    from perception.yolo_detector import YOLODetector
    from perception.midas_depth import MiDaSDepth
    from perception.depth_fusion import scale_midas_to_meters

    tof = ToFReader(range_mode_m=4)
    yolo = YOLODetector(model_path="yolov8n.pt", img_size=640)
    midas = MiDaSDepth(device="cpu")

    last_fit_time = 0.0
    cached_scale_depth = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        tof_m = tof.read_depth_m()  # (Ht, Wt) meters
        # TODO: align tof_m to frame resolution using calibration.
        # For a hackathon demo, you can resize:
        tof_aligned = cv2.resize(tof_m, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Near-field collision signal (0-4m)
        near_min = np.nanmin(tof_aligned)
        if np.isfinite(near_min) and near_min < 1.2:
            print(f"[WARN] Obstacle very close: {near_min:.2f}m")

        # Detections
        dets = yolo.detect(frame)

        # MiDaS depth -> meters (fit occasionally to save compute)
        now = time.time()
        if cached_scale_depth is None or (now - last_fit_time) > 1.0:
            d_rel = midas.predict(frame)
            try:
                d_m = scale_midas_to_meters(d_rel, tof_aligned)
                cached_scale_depth = d_m
                last_fit_time = now
            except Exception as e:
                print(f"[INFO] scale fit skipped: {e}")

        d_m = cached_scale_depth

        # Announce distance to interesting detections using median depth in bbox
        if d_m is not None:
            for det in dets:
                if det.conf < 0.35:
                    continue
                x1, y1, x2, y2 = det.xyxy
                crop = d_m[y1:y2, x1:x2]
                dist = np.nanmedian(crop) if crop.size else np.nan
                if np.isfinite(dist):
                    print(f"{det.cls:>12}  {det.conf:.2f}  ~{dist:.1f}m")

        # Basic UI (optional)
        cv2.imshow("rgb", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    tof.close()

if __name__ == "__main__":
    main()
