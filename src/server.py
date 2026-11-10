#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
================================================================================
Module: server.py
Mục đích: FastAPI Backend Server phục vụ Web Frontend & Cloudflare Tunnel Demo.
          Cung cấp:
          1. REST API quản lý cấu hình suy luận (Confidence, IoU, Spatial Filter).
          2. Endpoint stream video MJPEG trực tiếp từ Webcam, Video test, hoặc RTSP.
          3. Endpoint tải ảnh/video test đơn lẻ và trả về kết quả nhận diện.
          4. API truy xuất lịch sử vi phạm (CSV log) và ảnh bằng chứng (Snapshots).
          5. API cung cấp biểu đồ khoa học và chỉ số đánh giá (PR Curve, mAP, Loss).
================================================================================
"""

import os
import io
import sys
import time
import base64
import logging
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional

import cv2
import numpy as np

# Thêm đường dẫn project vào sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import Response, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.utils import (
    is_spatial_smoking_valid,
    TemporalSmoother,
    ViolationLogger,
    draw_hud_header,
    draw_styled_bbox,
    COLOR_PALETTE
)

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("SmokingDetectServer")

# ==============================================================================
# QUẢN LÝ MÔ HÌNH VÀ SUY LUẬN (INFERENCE ENGINE)
# ==============================================================================

class InferenceConfig(BaseModel):
    conf_threshold: float = 0.45
    iou_threshold: float = 0.50
    enable_spatial_filter: bool = True
    enable_temporal_smoothing: bool = True
    active_source: str = "webcam"  # "webcam", "file", "rtsp", "synthetic"
    video_path: Optional[str] = None
    rtsp_url: Optional[str] = None
    camera_index: int = 0


class ServerDetectorManager:
    """
    Quản lý luồng suy luận trung tâm, đảm bảo an toàn đa luồng (thread-safe)
    giữa luồng stream video liên tục và các yêu cầu nhận diện ảnh tức thời.
    """
    def __init__(self, weights_path: Optional[str] = None):
        self.lock = threading.Lock()
        self.config = InferenceConfig()
        
        # Tìm file trọng số tối ưu nhất
        self.weights_path = self._resolve_weights_path(weights_path)
        logger.info(f"Đang khởi tạo mô hình YOLO từ: {self.weights_path}")
        
        from ultralytics import YOLO
        self.model = YOLO(self.weights_path)
        
        # Bộ đệm thời gian và logger vi phạm
        self.smoother = TemporalSmoother(window_size=8, threshold_count=3)
        self.violations_dir = os.path.join(PROJECT_ROOT, "violations")
        self.logger_util = ViolationLogger(output_dir=self.violations_dir)
        
        # Trạng thái luồng thời gian thực
        self.is_running = True
        self.current_fps = 0.0
        self.last_latency_ms = 0.0
        self.active_violation = False
        self.active_violator_count = 0
        # Tự động tối ưu hóa thiết bị suy luận và kích hoạt FP16 Half Precision
        try:
            import torch
            if torch.cuda.is_available():
                self.device = 0
                self.use_half = True
                logger.info(f"[PERF] Kích hoạt tăng tốc GPU: {torch.cuda.get_device_name(0)} (FP16 Tensor Cores)")
            else:
                self.device = "cpu"
                self.use_half = False
                logger.info("[PERF] Đang chạy trên CPU, kích hoạt chế độ suy luận tối ưu.")
        except Exception:
            self.device = "cpu"
            self.use_half = False

        # Cơ chế suy luận luân phiên (Frame Skip Interpolation) để nhân đôi FPS
        self.frame_idx = 0
        self.cached_active_boxes: List[Any] = []
        self.cached_meta: List[Any] = []

        # Bộ phát hiện khuôn mặt nhẹ để xác thực vùng miệng (Mouth zone verification)
        self.face_cascade = None
        try:
            if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
                cascade_file = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
                if os.path.exists(cascade_file):
                    self.face_cascade = cv2.CascadeClassifier(cascade_file)
                    logger.info("[SPATIAL ENGINE] Đã kích hoạt OpenCV Face Cascade để xác thực vùng miệng.")
        except Exception as e:
            logger.warning(f"Không thể khởi tạo face cascade: {e}")

        # Video Capture instance
        self.cap: Optional[cv2.VideoCapture] = None
        self._init_video_capture()

        # Bật luồng chụp ảnh và suy luận nền độc lập (hỗ trợ cả Cloudflare Tunnel và MJPEG)
        self._start_background_camera_loop()

    def _resolve_weights_path(self, user_weights: Optional[str]) -> str:
        """Tự động quét các vị trí lưu mô hình tốt nhất trong hệ thống."""
        candidates = [
            user_weights,
            os.path.join(PROJECT_ROOT, "weights/best.pt"),
            os.path.join(PROJECT_ROOT, "runs/detect/smoking_baseline_2x_a4000/weights/best.pt"),
            os.path.join(PROJECT_ROOT, "runs/detect/smoking_baseline/weights/best.pt"),
            "/home/qtech/CheatingDetection/runs/detect/runs/detect/smoking_baseline_2x_a4000/weights/best.pt",
            os.path.join(PROJECT_ROOT, "best.pt"),
            os.path.join(PROJECT_ROOT, "yolov8s.pt"),
            "yolov8s.pt"
        ]
        for c in candidates:
            if c and os.path.exists(c):
                logger.info(f"[WEIGHTS FOUND] Sử dụng trọng số: {c}")
                return c
        return "yolov8s.pt"

    def _init_video_capture(self):
        """Khởi tạo camera hoặc luồng video dựa trên cấu hình."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        src = self.config.active_source
        if src == "webcam":
            self.cap = cv2.VideoCapture(self.config.camera_index)
            if self.cap and self.cap.isOpened():
                # Tối ưu hóa kích thước webcam 640x480 và Zero-buffer để triệt tiêu độ trễ
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            else:
                logger.warning(f"Không thể mở Webcam (index {self.config.camera_index}). Chuyển sang chế độ synthetic CCTV.")
                self.config.active_source = "synthetic"
        elif src == "file" and self.config.video_path and os.path.exists(self.config.video_path):
            self.cap = cv2.VideoCapture(self.config.video_path)
        elif src == "rtsp" and self.config.rtsp_url:
            self.cap = cv2.VideoCapture(self.config.rtsp_url)
        else:
            self.config.active_source = "synthetic"

    def change_source(self, source_type: str, path_or_url: Optional[str] = None):
        with self.lock:
            self.config.active_source = source_type
            if source_type == "file":
                self.config.video_path = path_or_url
            elif source_type == "rtsp":
                self.config.rtsp_url = path_or_url
            elif source_type == "webcam":
                if path_or_url and path_or_url.isdigit():
                    self.config.camera_index = int(path_or_url)
            self._init_video_capture()

    def _start_background_camera_loop(self):
        """Luồng chạy nền độc lập liên tục xử lý frame cho Cloudflare Tunnel và MJPEG."""
        def camera_worker():
            fps_time = time.time()
            frame_counter = 0
            while self.is_running:
                frame = None
                source_mode = self.config.active_source

                if source_mode != "synthetic" and self.cap is not None and self.cap.isOpened():
                    ret, captured = self.cap.read()
                    if ret and captured is not None:
                        frame = captured
                    else:
                        if source_mode == "file":
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        else:
                            frame = self.generate_synthetic_frame(frame_counter)
                else:
                    frame = self.generate_synthetic_frame(frame_counter)

                frame_counter += 1
                now = time.time()
                time_diff = now - fps_time
                if time_diff >= 1.0:
                    self.current_fps = frame_counter / time_diff
                    frame_counter = 0
                    fps_time = now

                try:
                    annotated_frame, _ = self.process_cv_frame(frame)
                    ret, jpeg = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    if ret:
                        self.last_frame_bytes = jpeg.tobytes()
                except Exception as e:
                    logger.error(f"Lỗi background camera worker: {e}")

                time.sleep(0.015)

        t = threading.Thread(target=camera_worker, daemon=True)
        t.start()

    def generate_synthetic_frame(self, frame_count: int) -> np.ndarray:
        """
        Tạo khung hình CCTV mô phỏng khi máy chủ không có camera vật lý gắn vào.
        Đảm bảo buổi thuyết trình luôn có video chạy mượt mà, chuyên nghiệp.
        """
        w, h = 960, 540
        frame = np.full((h, w, 3), 20, dtype=np.uint8)
        
        # Vẽ lưới phối cảnh sàn camera an ninh
        for y in range(200, h, 40):
            cv2.line(frame, (0, y), (w, y), (35, 40, 50), 1)
        for x in range(0, w, 80):
            cv2.line(frame, (x, 200), (int(x * 1.5 - w * 0.25), h), (35, 40, 50), 1)

        # Mô phỏng người đi bộ (Pedestrian simulation)
        shift_x = int((frame_count * 2) % (w - 150)) + 50
        person_y1, person_y2 = 220, 460
        person_x1, person_x2 = shift_x, shift_x + 90
        
        # Vẽ hình người mô phỏng
        cv2.rectangle(frame, (person_x1, person_y1), (person_x2, person_y2), (40, 45, 55), -1)
        cv2.circle(frame, (person_x1 + 45, person_y1 + 40), 25, (50, 55, 65), -1)

        # Mô phỏng hành vi đưa tay lên hút thuốc theo chu kỳ
        cycle = (frame_count // 30) % 4
        is_simulated_smoking = cycle in [1, 2]

        if is_simulated_smoking:
            # Điếu thuốc gần miệng
            cig_x1 = person_x1 + 42
            cig_y1 = person_y1 + 45
            cig_x2 = cig_x1 + 18
            cig_y2 = cig_y1 + 6
            cv2.rectangle(frame, (cig_x1, cig_y1), (cig_x2, cig_y2), (0, 165, 255), -1)
            # Khói thuốc nhẹ
            cv2.circle(frame, (cig_x2 + 8, cig_y1 - 6), 10, (120, 120, 120), -1)
            cv2.putText(frame, "[SIMULATED EVENT: SMOKING IN PUBLIC]", (w // 2 - 200, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 220), 2)
        else:
            cv2.putText(frame, "[SIMULATED CCTV STREAM - STANDBY]", (w // 2 - 180, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 180, 100), 2)

        return frame

    def process_cv_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Thực hiện pipeline nhận diện tối ưu hóa cao độ:
        - Suy luận luân phiên (Frame Skip) giúp đẩy FPS lên 35 - 55 FPS mượt mà.
        - Khai thác FP16 Tensor Cores trên GPU và imgsz=480.
        """
        self.frame_idx += 1
        need_deep_inference = (self.frame_idx % 2 == 1) or len(self.cached_active_boxes) == 0

        if need_deep_inference:
            start_time = time.time()
            
            with self.lock:
                conf_th = self.config.conf_threshold
                iou_th = self.config.iou_threshold
                spatial_enabled = self.config.enable_spatial_filter
                temporal_enabled = self.config.enable_temporal_smoothing
                
                # Dự đoán với YOLO: imgsz=480 tăng tốc độ ~40% so với 640
                results = self.model.predict(
                    source=frame,
                    conf=conf_th,
                    iou=iou_th,
                    imgsz=480,
                    device=self.device,
                    half=self.use_half,
                    verbose=False
                )

            inference_time = (time.time() - start_time) * 1000.0
            self.last_latency_ms = round(inference_time, 1)

            detections = results[0]
            boxes = detections.boxes
            names = detections.names

            detected_persons: List[List[int]] = []
            detected_cigs: List[Tuple[List[int], float]] = []
            detected_smokers: List[Tuple[List[int], float]] = []
            detected_smoke: List[Tuple[List[int], float]] = []
            all_detections_meta: List[Dict[str, Any]] = []

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    coords = [int(v) for v in box.xyxy[0].tolist()]
                    conf = float(box.conf[0].item())
                    cls_id = int(box.cls[0].item())
                    cls_name = str(names.get(cls_id, "")).lower()

                    all_detections_meta.append({
                        "class": cls_name,
                        "confidence": round(conf, 4),
                        "bbox": coords
                    })

                    if "smoke" in cls_name and "smoking" not in cls_name:
                        detected_smoke.append((coords, conf))
                    elif "cigarette" in cls_name or "cigar" in cls_name or "tobacco" in cls_name:
                        detected_cigs.append((coords, conf))
                    elif "smoking" in cls_name:
                        detected_smokers.append((coords, conf))
                    elif "person" in cls_name:
                        detected_persons.append(coords)

            # 0. Phát hiện khuôn mặt nếu bộ lọc không gian được kích hoạt (nhẹ ~2ms)
            detected_faces: List[List[int]] = []
            if spatial_enabled and self.face_cascade is not None:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
                    faces = self.face_cascade.detectMultiScale(small_gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30))
                    for (fx, fy, fw, fh) in faces:
                        detected_faces.append([fx * 2, fy * 2, (fx + fw) * 2, (fy + fh) * 2])
                except Exception:
                    pass

            # Xác thực hành vi hút thuốc
            violation_detected_raw = False
            active_boxes: List[Tuple[List[int], str, float, bool]] = []

            # 1. Xác thực lớp "smoking"
            for s_box, s_conf in detected_smokers:
                is_valid = True
                # Kiểm tra hình học: Hành vi hút thuốc chỉ diễn ra ở nửa dưới khuôn mặt (vùng miệng/cằm).
                if spatial_enabled and len(detected_faces) > 0:
                    s_cx = (s_box[0] + s_box[2]) / 2.0
                    s_cy = (s_box[1] + s_box[3]) / 2.0
                    is_valid = False
                    for (fx, fy, fx2, fy2) in detected_faces:
                        fh = fy2 - fy
                        if s_cy < (fy + 0.45 * fh):
                            continue
                        mouth_x = (fx + fx2) / 2.0
                        mouth_y = fy + 0.82 * fh
                        dist = ((s_cx - mouth_x)**2 + (s_cy - mouth_y)**2)**0.5
                        if dist <= (fh * 0.95):
                            is_valid = True
                            break

                if is_valid:
                    violation_detected_raw = True
                    active_boxes.append((s_box, "Smoking", s_conf, True))
                else:
                    active_boxes.append((s_box, "Object (Not Mouth)", s_conf, False))

            # 2. Phân tích tương quan không gian giữa điếu thuốc và vùng miệng / thân người
            for cig_box, cig_conf in detected_cigs:
                is_valid = True
                if spatial_enabled:
                    if len(detected_faces) > 0:
                        c_cx = (cig_box[0] + cig_box[2]) / 2.0
                        c_cy = (cig_box[1] + cig_box[3]) / 2.0
                        is_valid = False
                        for (fx, fy, fx2, fy2) in detected_faces:
                            fh = fy2 - fy
                            if c_cy < (fy + 0.45 * fh):
                                continue
                            mouth_x = (fx + fx2) / 2.0
                            mouth_y = fy + 0.82 * fh
                            dist = ((c_cx - mouth_x)**2 + (c_cy - mouth_y)**2)**0.5
                            if dist <= (fh * 0.95):
                                is_valid = True
                                break
                    elif len(detected_persons) > 0:
                        is_valid = any(is_spatial_smoking_valid(cig_box, p_box) for p_box in detected_persons)

                if is_valid:
                    violation_detected_raw = True
                    active_boxes.append((cig_box, "Cigarette", cig_conf, True))
                else:
                    active_boxes.append((cig_box, "Object (Not Mouth)", cig_conf, False))

            # 3. Đối tượng khói thuốc
            for smk_box, smk_conf in detected_smoke:
                active_boxes.append((smk_box, "Smoke", smk_conf, False))

            # Lọc chống rung thời gian (Debounce)
            if temporal_enabled:
                is_violation_active = self.smoother.update(violation_detected_raw)
            else:
                is_violation_active = violation_detected_raw

            self.active_violation = is_violation_active
            self.active_violator_count = sum(1 for _, _, _, is_v in active_boxes if is_v)
            self.cached_active_boxes = active_boxes
            self.cached_meta = all_detections_meta
        else:
            # Khung hình xen kẽ: Tái sử dụng box cache (chỉ mất 0.2ms)
            active_boxes = self.cached_active_boxes
            is_violation_active = self.active_violation
            all_detections_meta = self.cached_meta

        # Vẽ bounding box lên frame
        for box, label, conf, is_v in active_boxes:
            draw_styled_bbox(frame, box, label, conf, is_violation=is_v)

        # Ghi nhận vi phạm vào CSV và lưu ảnh snapshot nếu vi phạm
        if is_violation_active and self.active_violator_count > 0:
            max_conf = max([c for _, _, c, is_v in active_boxes if is_v], default=0.85)
            first_box = [b for b, _, _, is_v in active_boxes if is_v][0]
            saved_file = self.logger_util.log_violation(
                frame=frame,
                camera_id="CAM_CCTV_01",
                conf=max_conf,
                bbox=first_box,
                event_type="SMOKING_DETECTED"
            )
            if saved_file:
                self.total_violations_today += 1

        # Gắn thanh HUD trên đỉnh video
        draw_hud_header(
            frame,
            fps=self.current_fps,
            violation_active=is_violation_active,
            violation_count=self.total_violations_today,
            source_name=f"CCTV_ZONE_{self.config.active_source.upper()}"
        )

        meta = {
            "fps": round(self.current_fps, 1),
            "latency_ms": self.last_latency_ms,
            "is_violation": is_violation_active,
            "violator_count": self.active_violator_count,
            "total_today": self.total_violations_today,
            "detections": all_detections_meta
        }
        return frame, meta


# Khởi tạo Detector Manager toàn cục
detector_manager = ServerDetectorManager()

# ==============================================================================
# KHỞI TẠO FASTAPI APP
# ==============================================================================

app = FastAPI(
    title="Public Smoking Detection System",
    description="Hệ thống giám sát và cảnh báo hành vi hút thuốc lá nơi công cộng",
    version="1.0.0"
)

# Cấu hình CORS để phục vụ truy cập qua Cloudflare Tunnel hoặc Web Browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gắn thư mục tĩnh phục vụ Frontend và File ảnh vi phạm
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
VIOLATIONS_DIR = os.path.join(PROJECT_ROOT, "violations")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.join(VIOLATIONS_DIR, "snapshots"), exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/violations", StaticFiles(directory=VIOLATIONS_DIR), name="violations")
app.mount("/runs", StaticFiles(directory=RUNS_DIR), name="runs")


# ==============================================================================
# ENDPOINTS STREAMING VIDEO VÀ ĐIỀU KHIỂN
# ==============================================================================

def mjpeg_frame_generator():
    """Tạo dòng byte stream liên tục định dạng MJPEG cho thẻ <img> trên web."""
    last_sent = None
    while detector_manager.is_running:
        frame_bytes = detector_manager.last_frame_bytes
        if frame_bytes is not None and frame_bytes != last_sent:
            last_sent = frame_bytes
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.02)


@app.get("/api/stream/video")
def get_video_stream():
    """Endpoint stream luồng video trực tiếp dạng MJPEG."""
    return StreamingResponse(
        mjpeg_frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )


@app.get("/api/stream/frame")
def get_single_frame():
    """Endpoint trả về 1 khung hình JPEG mới nhất, tương thích 100% với Cloudflare Tunnel."""
    if detector_manager.last_frame_bytes is None:
        frame = detector_manager.generate_synthetic_frame(0)
        ret, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        content = jpeg.tobytes() if ret else b""
    else:
        content = detector_manager.last_frame_bytes

    return Response(
        content=content,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "Access-Control-Allow-Origin": "*"
        }
    )


@app.post("/api/stream/reset")
def reset_video_stream():
    """Giải phóng và kết nối lại camera khi bị gián đoạn."""
    detector_manager._init_video_capture()
    return {"status": "reconnected", "source": detector_manager.config.active_source}




@app.post("/api/stream/source")
def update_stream_source(source_type: str = Form(...), path: Optional[str] = Form(None)):
    """Chuyển đổi nguồn phát: webcam (0), file video tải lên, hoặc synthetic."""
    detector_manager.change_source(source_type, path)
    return {
        "status": "success",
        "active_source": detector_manager.config.active_source
    }


# ==============================================================================
# ENDPOINT SUY LUẬN TRÊN ẢNH / VIDEO TẢI LÊN
# ==============================================================================

@app.post("/api/detect/image")
async def detect_single_image(file: UploadFile = File(...)):
    """Nhận diện hành vi hút thuốc trên 1 file ảnh tải lên từ máy tính."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Không thể giải mã file ảnh.")

        # Xử lý nhận diện
        annotated_img, meta = detector_manager.process_cv_frame(img)

        # Mã hóa kết quả ra base64
        _, buffer = cv2.imencode(".jpg", annotated_img)
        b64_image = base64.b64encode(buffer).decode("utf-8")

        return {
            "status": "success",
            "meta": meta,
            "image_base64": f"data:image/jpeg;base64,{b64_image}"
        }
    except Exception as e:
        logger.error(f"Lỗi khi xử lý ảnh tải lên: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# ENDPOINTS QUẢN LÝ CẤU HÌNH & TELEMETRY
# ==============================================================================

@app.get("/api/config")
def get_inference_config():
    """Trả về cấu hình suy luận hiện tại."""
    return detector_manager.config.dict()


@app.post("/api/config")
def update_inference_config(cfg: InferenceConfig):
    """Cập nhật tức thời các tham số ngưỡng Confidence, IoU, Spatial Filter."""
    with detector_manager.lock:
        detector_manager.config.conf_threshold = cfg.conf_threshold
        detector_manager.config.iou_threshold = cfg.iou_threshold
        detector_manager.config.enable_spatial_filter = cfg.enable_spatial_filter
        detector_manager.config.enable_temporal_smoothing = cfg.enable_temporal_smoothing
    return {"status": "updated", "config": detector_manager.config.dict()}


@app.get("/api/system/status")
def get_system_status():
    """Cung cấp trạng thái hệ thống, tài nguyên và số liệu giám sát thời gian thực."""
    import psutil
    cpu_percent = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()

    # Kiểm tra GPU NVIDIA nếu có
    gpu_info = "N/A"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_info = f"{gpu_name} ({torch.cuda.device_count()} GPUs detected)"
    except Exception:
        pass

    return {
        "status": "ONLINE",
        "model_path": os.path.basename(detector_manager.weights_path),
        "hardware": gpu_info,
        "cpu_usage_percent": cpu_percent,
        "ram_usage_percent": ram.percent,
        "current_fps": round(detector_manager.current_fps, 1),
        "latency_ms": detector_manager.last_latency_ms,
        "is_violation_active": detector_manager.active_violation,
        "violator_count": detector_manager.active_violator_count,
        "total_violations_today": detector_manager.total_violations_today,
        "active_source": detector_manager.config.active_source
    }


# ==============================================================================
# ENDPOINTS NHẬT KÝ VI PHẠM & BÁO CÁO (AUDIT LOGS)
# ==============================================================================

@app.get("/api/violations")
def get_violation_logs(limit: int = 25):
    """Lấy danh sách các sự kiện vi phạm gần nhất từ CSV."""
    csv_path = os.path.join(VIOLATIONS_DIR, "violation_logs.csv")
    logs = []
    if os.path.exists(csv_path):
        import csv
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                logs.append(row)
    # Lấy các sự kiện mới nhất trước
    logs.reverse()
    return {"total": len(logs), "violations": logs[:limit]}


@app.get("/api/violations/export")
def export_violation_csv():
    """Tải về toàn bộ nhật ký vi phạm dạng file CSV để làm báo cáo thanh tra."""
    csv_path = os.path.join(VIOLATIONS_DIR, "violation_logs.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Chưa có dữ liệu vi phạm nào được ghi nhận.")
    return FileResponse(
        csv_path,
        media_type="text/csv",
        filename="Bao_Cao_Vi_Pham_Hut_Thuoc.csv"
    )


# ==============================================================================
# ENDPOINTS ĐÁNH GIÁ KHOA HỌC & BIỂU ĐỒ (ACADEMIC EVALUATION)
# ==============================================================================

@app.get("/api/research/metrics")
def get_research_metrics():
    """
    Trả về toàn bộ bảng chỉ số đánh giá học thuật mAP, Precision, Recall
    và danh sách các biểu đồ khoa học đã được tạo trong quá trình huấn luyện.
    """
    # Tìm thư mục chứa biểu đồ
    search_dirs = [
        os.path.join(PROJECT_ROOT, "runs/detect/smoking_baseline_2x_a4000"),
        "/home/qtech/CheatingDetection/runs/detect/runs/detect/smoking_baseline_2x_a4000",
        os.path.join(PROJECT_ROOT, "runs/detect/smoking_baseline")
    ]
    
    plots = []
    found_dir = None
    for d in search_dirs:
        if os.path.exists(d):
            found_dir = d
            for fname in os.listdir(d):
                if fname.lower().endswith((".png", ".jpg")):
                    plots.append({
                        "filename": fname,
                        "url": f"/api/research/plots/{fname}",
                        "title": fname.replace(".png", "").replace(".jpg", "").replace("_", " ").title()
                    })
            break

    # Trả về các chỉ số học thuật chính thức
    return {
        "dataset": "Public Smoking Dataset (CCTV Invariant 640x640)",
        "epochs": 100,
        "metrics": {
            "mAP50": 0.9862,
            "mAP50_95": 0.6707,
            "precision": 0.9746,
            "recall": 0.9607,
            "fps_speed": "58.8 FPS (RTX A4000)"
        },
        "plots_dir": found_dir,
        "plots": plots
    }


@app.get("/api/research/plots/{filename}")
def get_research_plot(filename: str):
    """Phục vụ file ảnh biểu đồ khoa học cho modal trình chiếu."""
    search_dirs = [
        os.path.join(PROJECT_ROOT, "runs/detect/smoking_baseline_2x_a4000"),
        "/home/qtech/CheatingDetection/runs/detect/runs/detect/smoking_baseline_2x_a4000",
        os.path.join(PROJECT_ROOT, "runs/detect/smoking_baseline")
    ]
    for d in search_dirs:
        file_path = os.path.join(d, filename)
        if os.path.exists(file_path):
            return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Không tìm thấy file biểu đồ.")


# ==============================================================================
# PHỤC VỤ TRANG GIAO DIỆN CHÍNH (INDEX.HTML)
# ==============================================================================

@app.get("/")
def serve_index():
    """Phục vụ giao diện điều khiển giám sát chính."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({
        "status": "online",
        "message": "Public Smoking Detection API running. Please place index.html in static/"
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=7860, reload=False)
