#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
================================================================================
Module: inference_demo.py
Mục đích: Chương trình mô phỏng xử lý luồng video / RTSP / webcam thời gian thực.
          Tích hợp:
          1. Nạp mô hình đã huấn luyện (best.pt / yolov8s.pt) với ngưỡng conf=0.45.
          2. Bộ lọc không gian (Spatial Check) giữa điếu thuốc và vùng thân trên/mặt
             nhằm triệt tiêu tối đa các cảnh báo giả (bút, que tăm, điện thoại).
          3. Bộ lọc thời gian (Temporal Smoothing) khử hiện tượng nháy cảnh báo.
          4. Hiển thị Heads-Up Display (HUD) cảnh báo đỏ 'CANH BAO: Hut thuoc la!'.
          5. Tự động ghi nhật ký vi phạm ra file CSV/TXT và chụp ảnh bằng chứng.
================================================================================
"""

import os
import sys
import time
import argparse
import logging
from typing import List, Dict, Any, Tuple, Optional

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None
    np = None

# Thêm đường dẫn project vào sys.path để import an toàn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
logger = logging.getLogger("InferenceDemo")


class SmokingDetectorEngine:
    """
    Động cơ suy luận (Inference Engine) cho bài toán phát hiện hành vi hút thuốc lá.
    """
    def __init__(
        self,
        weights_path: str,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.50,
        device: str = "0",
        camera_id: str = "CAM_CCTV_01",
        violations_dir: str = "violations"
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.camera_id = camera_id
        
        # 1. Tải mô hình YOLO
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("Vui lòng cài đặt ultralytics: pip install ultralytics")
            sys.exit(1)

        if not os.path.exists(weights_path):
            logger.warning(f"Không tìm thấy file trọng số '{weights_path}'. Sẽ tự động tải yolov8s.pt để chạy demo.")
            weights_path = "yolov8s.pt"

        logger.info(f"Đang nạp mô hình từ: {weights_path}")
        self.model = YOLO(weights_path)
        self.device = device

        # 2. Khởi tạo bộ lọc thời gian và Logger
        self.smoother = TemporalSmoother(window_size=8, threshold_count=3)
        self.logger = ViolationLogger(output_dir=violations_dir)
        self.total_violation_events = 0

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool, int]:
        """
        Xử lý 1 khung hình đơn lẻ:
        - Chạy Object Detection
        - Phân tích tương quan không gian (Spatial Verification)
        - Cập nhật bộ đệm thời gian
        - Vẽ HUD cảnh báo và lưu log nếu vi phạm
        
        Returns:
            Tuple[annotated_frame, is_violation_active, num_violators]
        """
        h, w = frame.shape[:2]

        # 1. Thực hiện suy luận (Inference)
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )

        detections = results[0]
        boxes = detections.boxes

        # Danh sách chứa các đối tượng phát hiện
        detected_persons: List[List[int]] = []
        detected_cigarettes: List[Tuple[List[int], float]] = []
        detected_smokers: List[Tuple[List[int], float]] = []
        detected_smoke: List[Tuple[List[int], float]] = []

        # Tên các lớp trong mô hình
        names = detections.names

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                coords = [int(v) for v in box.xyxy[0].tolist()]
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                cls_name = str(names.get(cls_id, "")).lower()

                # Phân loại thực thể dựa trên nhãn
                if "smoke" in cls_name and "smoking" not in cls_name:
                    detected_smoke.append((coords, conf))
                elif "cigarette" in cls_name or "cigar" in cls_name or "tobacco" in cls_name:
                    detected_cigarettes.append((coords, conf))
                elif "smoking" in cls_name:
                    detected_smokers.append((coords, conf))
                elif "person" in cls_name:
                    detected_persons.append(coords)
                else:
                    # Nếu bộ dữ liệu gán nhãn tùy biến, xem xét cls_id
                    # Mặc định: class 0 là smoking hoặc person
                    if cls_id == 0 and "person" in cls_name:
                        detected_persons.append(coords)
                    elif cls_id == 1 or "smoking" in cls_name:
                        detected_smokers.append((coords, conf))

        # 2. Logic Hậu Xử Lý & Xác thực Không Gian (Spatial Reasoning)
        # Trường hợp A: Mô hình phát hiện trực tiếp hành vi 'smoking' (người đang hút thuốc)
        current_violators: List[Tuple[List[int], float, str]] = []

        for box, conf in detected_smokers:
            current_violators.append((box, conf, "SMOKING_BEHAVIOR"))

        # Trường hợp B: Mô hình phát hiện 'cigarette' hoặc 'smoke' độc lập
        # Cần kiểm tra xem nó có nằm gần thân trên/đầu/mặt của 'person' hay không
        for cig_box, cig_conf in detected_cigarettes:
            matched_person = False
            for p_box in detected_persons:
                if is_spatial_smoking_valid(cig_box, p_box, distance_threshold_ratio=0.50):
                    # Khẳng định người này đang hút thuốc
                    current_violators.append((p_box, cig_conf, "CIGARETTE_IN_HAND_FACE"))
                    matched_person = True
                    break
            
            # Nếu không tìm thấy người nào gần điếu thuốc, có thể là tàn thuốc rơi trên bàn/sàn,
            # hoặc vật thể hình trụ bị nhận diện sai -> Không kích hoạt báo động sai.
            if not matched_person and not detected_persons:
                # Nếu không có class person trong model nhưng phát hiện cigarette với conf rất cao
                current_violators.append((cig_box, cig_conf, "CIGARETTE_DETECTED"))

        # 3. Lọc qua bộ đệm thời gian (Temporal Smoothing)
        raw_violation = len(current_violators) > 0
        smooth_violation = self.smoother.update(raw_violation)

        # 4. Ghi nhận vi phạm & Bằng chứng (Logger & Snapshot)
        if smooth_violation and current_violators:
            top_violator = current_violators[0]
            snapshot = self.logger.log_violation(
                frame=frame,
                camera_id=self.camera_id,
                conf=top_violator[1],
                bbox=top_violator[0],
                event_type=top_violator[2]
            )
            if snapshot:
                self.total_violation_events += 1
                logger.warning(f"[CANH BAO] Phat hien hanh vi hut thuoc tai {self.camera_id}! Anh bang chung: {snapshot}")

        # 5. Vẽ Bounding Boxes và Giao diện HUD
        annotated_frame = frame.copy()

        # Vẽ người bình thường (màu xanh lá)
        for p_box in detected_persons:
            # Chỉ vẽ nếu người đó không nằm trong danh sách vi phạm
            is_violator = any(p_box == v[0] for v in current_violators)
            if not is_violator:
                draw_styled_bbox(annotated_frame, p_box, "Person", 0.9, is_violation=False)

        # Vẽ các điếu thuốc (màu vàng)
        for cig_box, c_conf in detected_cigarettes:
            draw_styled_bbox(annotated_frame, cig_box, "Cigarette", c_conf, is_violation=True)

        # Vẽ người vi phạm hút thuốc (màu đỏ rực rỡ)
        for v_box, v_conf, v_type in current_violators:
            draw_styled_bbox(annotated_frame, v_box, "CANH BAO: Hut Thuoc", v_conf, is_violation=True)

        return (annotated_frame, smooth_violation, len(current_violators))


def run_inference_stream(
    source: str,
    weights: str,
    conf: float = 0.45,
    iou: float = 0.50,
    device: str = "0",
    output_path: Optional[str] = None,
    no_show: bool = False,
    camera_id: str = "CCTV_PUBLIC_CAM01"
):
    """
    Vòng lặp chính xử lý luồng Video / RTSP / Webcam.
    """
    if not CV2_AVAILABLE:
        logger.error("Thư viện 'opencv-python' chưa được cài đặt.")
        print("[HƯỚNG DẪN] Vui lòng chạy: pip install opencv-python")
        return

    # Khởi tạo động cơ phát hiện
    engine = SmokingDetectorEngine(
        weights_path=weights,
        conf_threshold=conf,
        iou_threshold=iou,
        device=device,
        camera_id=camera_id
    )

    # Mở nguồn video
    # Kiểm tra nếu là webcam dạng số nguyên (ví dụ: '0', '1')
    stream_source: Any = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(stream_source)

    if not cap.isOpened():
        logger.error(f"Không thể mở nguồn video: {source}")
        print("[GỢI Ý] Hãy kiểm tra lại đường dẫn file video hoặc quyền truy cập Webcam/RTSP.")
        return

    # Lấy thông số luồng video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    logger.info(f"Đã kết nối thành công nguồn phát: {source}")
    logger.info(f"Độ phân giải: {frame_width}x{frame_height} @ {source_fps:.1f} FPS")

    # Bộ ghi video (VideoWriter) nếu có yêu cầu lưu file
    writer = None
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, source_fps, (frame_width, frame_height))
        logger.info(f"Kết quả phân tích sẽ được ghi vào: {output_path}")

    fps_meter = 0.0
    frame_count = 0
    start_time = time.time()

    print("\n" + "=" * 65)
    print(" HỆ THỐNG GIÁM SÁT & CẢNH BÁO HÚT THUỐC LÁ ĐANG HOẠT ĐỘNG")
    print(f"  - Nguồn Stream       : {source}")
    print(f"  - Ngưỡng Tin Cậy     : {conf}")
    print(f"  - Camera ID          : {camera_id}")
    print("  - Phím điều khiển    : Bấm 'q' trên cửa sổ để thoát")
    print("=" * 65 + "\n")

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                logger.info("Luồng video kết thúc hoặc không còn dữ liệu.")
                break

            frame_count += 1

            # Xử lý khung hình với động cơ AI
            annotated_frame, is_violation, num_violators = engine.process_frame(frame)

            # Tính toán FPS hiển thị
            t_elapsed = time.time() - t0
            current_fps = 1.0 / t_elapsed if t_elapsed > 0 else 30.0
            fps_meter = 0.9 * fps_meter + 0.1 * current_fps if fps_meter > 0 else current_fps

            # Vẽ thanh trạng thái Header HUD
            final_frame = draw_hud_header(
                annotated_frame,
                fps=fps_meter,
                violation_active=is_violation,
                violation_count=engine.total_violation_events,
                source_name=camera_id
            )

            # Lưu vào video đầu ra
            if writer:
                writer.write(final_frame)

            # Hiển thị lên màn hình (nếu không ở chế độ headless)
            if not no_show:
                cv2.imshow("Smoking Detection & Alert System (AI Monitor)", final_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Phím q hoặc ESC
                    logger.info("Nhận lệnh dừng từ người dùng.")
                    break

    except KeyboardInterrupt:
        logger.info("Nhận tín hiệu ngắt (Ctrl+C). Đang đóng tiến trình an toàn...")
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        total_time = time.time() - start_time
        logger.info(f"Hoàn tất! Tổng khung hình đã xử lý: {frame_count} ({frame_count/max(1, total_time):.1f} FPS TB).")
        logger.info(f"Tổng số sự kiện vi phạm được ghi lại: {engine.total_violation_events}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Real-time Smoking Detection & Warning System for Public CCTV"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Nguồn video: Đường dẫn file .mp4, số camera webcam (0, 1) hoặc URL RTSP."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/smoking_baseline_2x_a4000/weights/best.pt",
        help="Đường dẫn file trọng số mô hình đã huấn luyện (best.pt)."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="Ngưỡng tin cậy tối thiểu (confidence threshold, mặc định 0.45)."
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.50,
        help="Ngưỡng NMS IoU threshold (mặc định 0.50)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Thiết bị GPU chạy suy luận ('0' hoặc 'cpu')."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Đường dẫn lưu file video kết quả (ví dụ: output_alert.mp4)."
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Chạy ở chế độ Headless (không mở cửa sổ đồ họa, tối ưu khi chạy trên server qua SSH)."
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default="CAM_PUBLIC_01",
        help="Mã định danh camera giám sát phục vụ việc ghi nhật ký vi phạm."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_inference_stream(
        source=args.source,
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        output_path=args.output,
        no_show=args.no_show,
        camera_id=args.camera_id
    )
