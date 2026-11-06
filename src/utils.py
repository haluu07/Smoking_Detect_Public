#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
================================================================================
Module: utils.py
Mục đích: Cung cấp các hàm tiện ích hình học (Spatial Geometric Verification), 
          xử lý thời gian (Temporal Smoothing), hiển thị giao diện đồ họa (HUD),
          và ghi nhật ký vi phạm (Violation Logger) phục vụ bài toán giám sát.
================================================================================
"""

import os
import csv
import time
import math
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None
    np = None


# ==============================================================================
# 1. HÌNH HỌC VÀ QUAN HỆ KHÔNG GIAN (SPATIAL GEOMETRIC CHECKS)
# ==============================================================================

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Tính chỉ số Intersection over Union (IoU) giữa 2 bounding boxes định dạng [x1, y1, x2, y2].
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        float: Giá trị IoU trong đoạn [0.0, 1.0]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    
    union_area = area1 + area2 - intersection_area
    if union_area <= 0.0:
        return 0.0
    return intersection_area / union_area


def get_box_center(box: List[float]) -> Tuple[float, float]:
    """Trả về tọa độ tâm (center_x, center_y) của bounding box [x1, y1, x2, y2]."""
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def calculate_euclidean_distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
    """Tính khoảng cách Euclid giữa 2 điểm trong không gian pixel 2D."""
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def is_spatial_smoking_valid(
    cig_box: List[float],
    person_box: List[float],
    distance_threshold_ratio: float = 0.55
) -> bool:
    """
    Kiểm tra quan hệ không gian (Spatial Heuristic Check) giữa điếu thuốc (cigarette)
    hoặc khói (smoke) với người (person) để lọc bỏ báo động giả (False Positives).
    
    Quy tắc nghiệp vụ:
    - Điếu thuốc phải nằm trong vùng lân cận thân trên hoặc gần đầu/mặt của người.
    - Giả định vùng đầu/mặt của người chiếm khoảng 35-40% chiều cao phía trên của person_box.
    - Khoảng cách từ tâm điếu thuốc tới vùng thân trên không được vượt quá ngưỡng tương đối
      so với kích thước của người (person height/width).

    Args:
        cig_box: [x1, y1, x2, y2] của điếu thuốc / khói
        person_box: [x1, y1, x2, y2] của người
        distance_threshold_ratio: Tỷ lệ ngưỡng khoảng cách tương đối so với chiều cao người

    Returns:
        bool: True nếu điếu thuốc có vị trí hợp lý với tư thế hút thuốc của người.
    """
    px1, py1, px2, py2 = person_box
    cx1, cy1, cx2, cy2 = cig_box

    person_w = max(1.0, px2 - px1)
    person_h = max(1.0, py2 - py1)

    # Vùng mặt ước lượng: Nửa trên của người (từ py1 đến py1 + 0.45 * person_h)
    face_region_y_max = py1 + 0.45 * person_h
    face_center_x = (px1 + px2) / 2.0
    face_center_y = py1 + (0.22 * person_h)

    cig_cx, cig_cy = get_box_center(cig_box)

    # 1. Kiểm tra bao hàm: Nếu điếu thuốc nằm hoàn toàn bên trong vùng thân trên
    if (px1 - 0.1 * person_w <= cig_cx <= px2 + 0.1 * person_w) and (py1 - 0.1 * person_h <= cig_cy <= face_region_y_max):
        return True

    # 2. Kiểm tra khoảng cách tâm tới vị trí mặt ước lượng
    distance_to_face = calculate_euclidean_distance((cig_cx, cig_cy), (face_center_x, face_center_y))
    threshold_dist = person_h * distance_threshold_ratio

    return distance_to_face <= threshold_dist


# ==============================================================================
# 2. XỬ LÝ THỜI GIAN VÀ CHỐNG RUNG CẢNH BÁO (TEMPORAL FILTERING)
# ==============================================================================

class TemporalSmoother:
    """
    Bộ đệm chống rung cảnh báo (Debounce / Temporal Moving Window):
    Chỉ kích hoạt cảnh báo khi hành vi hút thuốc được phát hiện trong ít nhất K 
    khung hình trên tổng số N khung hình gần nhất.
    """
    def __init__(self, window_size: int = 10, threshold_count: int = 4):
        self.window_size = window_size
        self.threshold_count = threshold_count
        self.history: List[bool] = []

    def update(self, detected: bool) -> bool:
        """Cập nhật trạng thái khung hình mới và trả về trạng thái cảnh báo sau lọc."""
        self.history.append(detected)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        positive_count = sum(self.history)
        return positive_count >= self.threshold_count

    def reset(self):
        self.history.clear()


# ==============================================================================
# 3. GIAO DIỆN HIỂN THỊ (HEADS-UP DISPLAY - HUD & OVERLAY)
# ==============================================================================

# Bảng màu trực quan (BGR)
COLOR_PALETTE = {
    'alert_red': (0, 0, 230),       # Cảnh báo vi phạm
    'warning_yellow': (0, 215, 255),# Vật thể nghi vấn (Cigarette)
    'smoke_gray': (180, 180, 180),  # Khói (Smoke)
    'safe_green': (50, 205, 50),    # Người bình thường
    'hud_dark': (25, 25, 25),       # Nền HUD đen mờ
    'white': (255, 255, 255),
}


def draw_rounded_rectangle(
    img: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    radius: int = 8
) -> np.ndarray:
    """Vẽ hình chữ nhật bo góc mềm mại cho giao diện camera hiện đại."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    # 4 cạnh
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)

    # 4 góc cong
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    return img


def draw_hud_header(
    frame: np.ndarray,
    fps: float,
    violation_active: bool,
    violation_count: int,
    source_name: str = "CAM_PUBLIC_01"
) -> np.ndarray:
    """
    Vẽ thanh trạng thái phía trên màn hình (Top Header HUD) với thông tin giám sát thời gian thực.
    """
    h, w = frame.shape[:2]
    header_h = 55

    # Tạo lớp phủ đen bán trong suốt (semi-transparent overlay)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), COLOR_PALETTE['hud_dark'], -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Thông tin Camera & FPS
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info_text = f"[{source_name}] | {timestamp_str} | FPS: {fps:.1f} | VIOLATIONS: {violation_count}"
    cv2.putText(frame, info_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_PALETTE['white'], 2)

    # Nếu đang có vi phạm, nhấp nháy hoặc hiển thị banner đỏ rực rỡ
    if violation_active:
        banner_w = 420
        banner_x1 = max(10, w - banner_w - 20)
        banner_y1 = 8
        banner_x2 = w - 20
        banner_y2 = header_h - 8

        # Khối màu đỏ nổi bật
        cv2.rectangle(frame, (banner_x1, banner_y1), (banner_x2, banner_y2), COLOR_PALETTE['alert_red'], -1)
        alert_msg = "!! CANH BAO: HUT THUOC LA !!"
        cv2.putText(frame, alert_msg, (banner_x1 + 15, banner_y1 + 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, COLOR_PALETTE['white'], 2, cv2.LINE_AA)

    return frame


def draw_styled_bbox(
    frame: np.ndarray,
    box: List[int],
    label: str,
    conf: float,
    is_violation: bool = False
) -> np.ndarray:
    """
    Vẽ bounding box phong cách chuyên nghiệp, kèm nhãn hiển thị có nền sắc nét.
    """
    x1, y1, x2, y2 = box
    color = COLOR_PALETTE['alert_red'] if is_violation else COLOR_PALETTE['warning_yellow']
    thickness = 3 if is_violation else 2

    # Vẽ khung bo góc
    draw_rounded_rectangle(frame, (x1, y1), (x2, y2), color, thickness=thickness, radius=6)

    # Nhãn hiển thị
    text = f"{label} {conf:.2f}"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

    # Nền nhãn
    badge_y1 = max(0, y1 - th - 10)
    badge_y2 = y1
    cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 10, badge_y2), color, -1)
    cv2.putText(frame, text, (x1 + 5, badge_y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_PALETTE['white'], 1, cv2.LINE_AA)

    return frame


# ==============================================================================
# 4. GHI NHẬT KÝ VÀ BẰNG CHỨNG VI PHẠM (LOGGER & SNAPSHOT RECORDER)
# ==============================================================================

class ViolationLogger:
    """
    Ghi nhận vi phạm ra file CSV và lưu ảnh bằng chứng (Snapshot) kèm watermark.
    """
    def __init__(self, output_dir: str = "violations", log_filename: str = "violation_logs.csv"):
        self.output_dir = output_dir
        self.snapshot_dir = os.path.join(output_dir, "snapshots")
        self.csv_path = os.path.join(output_dir, log_filename)

        os.makedirs(self.snapshot_dir, exist_ok=True)
        self._init_csv()

        # Cooldown để không chụp hàng trăm ảnh trong cùng 1 sự việc (tính bằng giây)
        self.last_log_time: float = 0.0
        self.cooldown_seconds: float = 3.0

    def _init_csv(self):
        """Khởi tạo tiêu đề cột cho file CSV nếu chưa tồn tại."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "Camera_ID", "Confidence", 
                    "BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2", 
                    "Event_Type", "Snapshot_File"
                ])

    def log_violation(
        self,
        frame: np.ndarray,
        camera_id: str,
        conf: float,
        bbox: List[int],
        event_type: str = "SMOKING_DETECTED"
    ) -> Optional[str]:
        """
        Ghi một dòng sự kiện vi phạm và lưu ảnh chụp bằng chứng nếu hết thời gian cooldown.
        """
        current_time = time.time()
        if (current_time - self.last_log_time) < self.cooldown_seconds:
            return None  # Đang trong thời gian cooldown

        self.last_log_time = current_time
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        filename_time = now.strftime("%Y%m%d_%H%M%S_%f")[:19]
        snapshot_filename = f"violation_{camera_id}_{filename_time}.jpg"
        snapshot_full_path = os.path.join(self.snapshot_dir, snapshot_filename)

        # Đóng watermark thời gian lên ảnh snapshot
        evidence_img = frame.copy()
        cv2.putText(
            evidence_img,
            f"EVIDENCE | {timestamp_str} | CONF: {conf:.2f}",
            (30, evidence_img.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA
        )
        cv2.imwrite(snapshot_full_path, evidence_img)

        # Ghi vào CSV
        with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp_str, camera_id, f"{conf:.4f}",
                bbox[0], bbox[1], bbox[2], bbox[3],
                event_type, snapshot_filename
            ])

        return snapshot_full_path
