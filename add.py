"""
Streamlit + YOLO11 + OpenCV demo app

Run locally:
  pip install -U streamlit ultralytics opencv-python numpy
  streamlit run app.py

Notes:
- Defaults to the YOLO11 nano weights ("yolo11n.pt"). Ultralytics will auto-download on first run.
- Webcam mode only works on local environments that can access your camera. On Streamlit Cloud, use image/video upload modes.
- You can optionally provide a custom model file (.pt or .onnx) via the sidebar.
"""
from __future__ import annotations

import time
from io import BytesIO
from typing import List, Optional

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(
    page_title="YOLO11 + OpenCV (Streamlit)",
    page_icon="🦾",
    layout="wide",
)

# ------------------------------
# Utilities
# ------------------------------
@st.cache_resource(show_spinner=False)
def load_model(weights_path: str) -> YOLO:
    """Load YOLO model once and cache it."""
    model = YOLO(weights_path)
    return model


def draw_info_bar(frame: np.ndarray, text: str, alpha: float = 0.6) -> np.ndarray:
    """Overlay an info bar with semi-transparent background at the top of the frame."""
    h, w = frame.shape[:2]
    bar_h = max(32, h // 18)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.putText(out, text, (12, int(bar_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def results_to_table(res) -> List[dict]:
    rows = []
    if not hasattr(res, "boxes") or res.boxes is None:
        return rows
    names = res.names
    for b in res.boxes:
        cls = int(b.cls.item())
        conf = float(b.conf.item()) if hasattr(b, "conf") else None
        xyxy = b.xyxy[0].tolist()  # [x1,y1,x2,y2]
        rows.append({
            "class_id": cls,
            "class_name": names.get(cls, str(cls)) if isinstance(names, dict) else str(cls),
            "confidence": round(conf, 4) if conf is not None else None,
            "x1": round(xyxy[0], 1),
            "y1": round(xyxy[1], 1),
            "x2": round(xyxy[2], 1),
            "y2": round(xyxy[3], 1),
        })
    return rows


# ------------------------------
# Sidebar controls
# ------------------------------
st.sidebar.header("⚙️ Settings")
model_source = st.sidebar.selectbox("Model weights", ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt", "Custom (.pt/.onnx)"])
custom_model_file = None
if model_source == "Custom (.pt/.onnx)":
    custom_model_file = st.sidebar.file_uploader("Upload custom YOLO model", type=["pt", "onnx"])

conf_thres = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.5, 0.01)
iou_thres = st.sidebar.slider("NMS IoU threshold", 0.1, 0.9, 0.5, 0.01)
max_det = st.sidebar.slider("Max detections per image", 10, 300, 100, 10)

source_mode = st.sidebar.radio("Source", ["Image", "Video file", "Webcam (local)"])

# Load model
weights_path = "yolo11n.pt" if model_source != "Custom (.pt/.onnx)" else None
if model_source != "Custom (.pt/.onnx)":
    weights_path = model_source
else:
    if custom_model_file is None:
        st.sidebar.info("Upload your custom weights to proceed.")
    else:
        # Persist uploaded file to tmp and load from path because ultralytics expects a path/str
        with open("uploaded_model.weights", "wb") as f:
            f.write(custom_model_file.read())
        weights_path = "uploaded_model.weights"

if weights_path is None:
    st.stop()

with st.sidebar:
    with st.spinner("Loading model..."):
        try:
            model = load_model(weights_path)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

# Class filter (after model loads)
model_names = model.names if hasattr(model, "names") else {}
all_class_names = [model_names[i] for i in sorted(model_names.keys())] if isinstance(model_names, dict) else []
class_filter: Optional[List[str]] = None
if all_class_names:
    class_filter = st.sidebar.multiselect("Filter classes (optional)", options=all_class_names)

# Speed/quality toggle
profile = st.sidebar.selectbox("Speed vs Quality", ["Balanced", "Fast", "Quality"])
if profile == "Fast":
    imgsz = 640
    vid_stride = 2
elif profile == "Quality":
    imgsz = 960
    vid_stride = 1
else:
    imgsz = 768
    vid_stride = 1

# ------------------------------
# Main UI
# ------------------------------
st.title("🦾 YOLO11 Object Detection (Streamlit + OpenCV)")
st.caption("Upload an image/video or use your webcam locally. Adjust thresholds in the sidebar.")

# Helper: run inference on a single frame (numpy array, BGR)
def infer_and_plot(frame_bgr: np.ndarray):
    results = model.predict(
        source=frame_bgr[..., ::-1],  # supply RGB
        conf=conf_thres,
        iou=iou_thres,
        imgsz=imgsz,
        max_det=max_det,
        classes=[all_class_names.index(c) for c in class_filter] if class_filter else None,
        verbose=False,
    )
    res = results[0]
    plotted = res.plot()  # returns RGB
    table = results_to_table(res)
    return plotted, table, res

if source_mode == "Image":
    col_u, col_res = st.columns([1, 1])
    img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if img_file is not None:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Could not decode the image.")
            st.stop()
        rgb_annot, table, res = infer_and_plot(bgr)
        with col_u:
            st.subheader("Original")
            st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_column_width=True)
        with col_res:
            st.subheader("Detections")
            st.image(rgb_annot, use_column_width=True)
        st.divider()
        st.subheader("Detections table")
        st.dataframe(table, use_container_width=True)
    else:
        st.info("👈 Upload an image to start.")

elif source_mode == "Video file":
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv", "webm"])
    if video_file is not None:
        # Save to temp file for OpenCV
        tmp_path = "uploaded_video.mp4"
        with open(tmp_path, "wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            st.error("Failed to open the video.")
            st.stop()
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        st.write(f"Resolution: {width}×{height} | FPS: {fps:.2f} | Frames: {total}")
        ph = st.empty()
        pbar = st.progress(0)
        stop = st.button("⏹ Stop")

        frame_idx = 0
        t0 = time.time()
        while cap.isOpened():
            if stop:
                break
            ret, frame = cap.read()
            if not ret:
                break
            if vid_stride > 1 and (frame_idx % vid_stride != 0):
                frame_idx += 1
                continue

            rgb_annot, table, res = infer_and_plot(frame)
            # Draw info bar
            elapsed = time.time() - t0
            disp = rgb_annot
            if elapsed > 0:
                cur_fps = (frame_idx + 1) / max(elapsed, 1e-6)
                disp_bgr = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
                disp_bgr = draw_info_bar(disp_bgr, f"FPS ~ {cur_fps:.1f} | detections: {len(table)}")
                disp = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)

            ph.image(disp, use_column_width=True)
            if total > 0:
                pbar.progress(min((frame_idx + 1) / total, 1.0))
            frame_idx += 1
        cap.release()
        st.success("Done processing video.")
    else:
        st.info("👈 Upload a video to start.")

else:  # Webcam (local)
    st.warning("Webcam mode is intended for local runs. If it doesn't show video, run this app locally.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not access the webcam.")
        st.stop()
    ph = st.empty()
    stop = st.button("⏹ Stop webcam")
    frame_idx = 0
    t0 = time.time()
    while True:
        if stop:
            break
        ok, frame = cap.read()
        if not ok:
            st.error("Failed to read from webcam.")
            break
        rgb_annot, table, res = infer_and_plot(frame)
        elapsed = time.time() - t0
        if elapsed > 0:
            cur_fps = (frame_idx + 1) / max(elapsed, 1e-6)
            disp_bgr = cv2.cvtColor(rgb_annot, cv2.COLOR_RGB2BGR)
            disp_bgr = draw_info_bar(disp_bgr, f"FPS ~ {cur_fps:.1f} | detections: {len(table)}")
            rgb_annot = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)
        ph.image(rgb_annot, use_column_width=True)
        frame_idx += 1
    cap.release()
    st.success("Webcam stopped.")
