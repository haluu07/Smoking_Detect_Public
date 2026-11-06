#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
================================================================================
Module: data_preprocessing.py
Mục đích: 
  1. Quét và kiểm tra tính toàn vẹn của dataset (ảnh hỏng, file mồ côi, bbox sai cú pháp).
  2. Chuẩn hóa và làm sạch tọa độ YOLO bounding box [class_id, x_c, y_c, w, h].
  3. Tự động chia tập train/val theo tỷ lệ chuẩn (80/20) với random seed tái lập.
  4. Tăng cường dữ liệu (Data Augmentation) với Albumentations:
     - Mô phỏng biến thiên ánh sáng (ColorJitter, CLAHE, RandomBrightnessContrast).
     - Mô phỏng camera CCTV rung lắc, người di chuyển (MotionBlur, GaussianBlur).
     - Mô phỏng góc quay camera giám sát trên cao (Perspective, ShiftScaleRotate).
     - Mô phỏng che khuất vật thể (CoarseDropout/Cutout).
================================================================================
"""

import os
import glob
import random
import shutil
import logging
import argparse
from typing import List, Tuple, Dict, Any, Optional

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None
    np = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("DataPreprocessing")

# Các đuôi file ảnh được hỗ trợ
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ==============================================================================
# 1. KIỂM TRA TÍNH TOÀN VẸN VÀ CHUẨN HÓA BOUNDING BOX (INTEGRITY VALIDATION)
# ==============================================================================

def validate_image_file(image_path: str) -> Optional[Tuple[int, int]]:
    """
    Kiểm tra xem file ảnh có đọc được không và không bị hỏng cấu trúc.
    
    Returns:
        Tuple[int, int]: (width, height) nếu ảnh tốt, None nếu ảnh bị lỗi/corrupt.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()  # Kiểm tra tính toàn vẹn của file
            
        # Mở lại để lấy kích thước chính xác
        with Image.open(image_path) as img:
            w, h = img.size
            if w <= 0 or h <= 0:
                return None
            return (w, h)
    except Exception as e:
        logger.debug(f"Ảnh lỗi {image_path}: {e}")
        return None


def validate_and_clean_yolo_labels(
    label_path: str,
    allow_empty: bool = False
) -> Tuple[bool, List[List[float]], List[int]]:
    """
    Kiểm tra và chuẩn hóa nội dung file nhãn định dạng YOLO (.txt):
    Mỗi dòng: class_id x_center y_center width height (tất cả chuẩn hóa 0.0 -> 1.0)
    
    Args:
        label_path: Đường dẫn file .txt
        allow_empty: Cho phép file rỗng (dành cho negative samples)
        
    Returns:
        Tuple[is_valid, bboxes_normalized, class_ids]
    """
    if not os.path.exists(label_path):
        return (False, [], [])

    cleaned_bboxes = []
    class_ids = []

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if not lines:
            # File rỗng (Negative sample - không có vật thể hút thuốc)
            return (allow_empty, [], [])

        for line_num, line in enumerate(lines, 1):
            parts = line.split()
            if len(parts) != 5:
                logger.warning(f"Dòng {line_num} trong {label_path} không đủ 5 trường: '{line}'")
                continue

            try:
                cls_id = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                logger.warning(f"Lỗi ép kiểu số tại dòng {line_num} file {label_path}")
                continue

            # Kiểm tra và tự động kẹp (clip) dung sai nhỏ nếu xém ngoài biên [0, 1]
            eps = 1e-4
            if not (-eps <= xc <= 1.0 + eps and -eps <= yc <= 1.0 + eps and 0.0 < w <= 1.0 + eps and 0.0 < h <= 1.0 + eps):
                logger.warning(f"Tọa độ bbox ngoài khoảng [0, 1] tại {label_path} dòng {line_num}: {parts}")
                continue

            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            w = max(0.001, min(1.0, w))
            h = max(0.001, min(1.0, h))

            cleaned_bboxes.append([xc, yc, w, h])
            class_ids.append(cls_id)

        if not cleaned_bboxes and not allow_empty:
            return (False, [], [])

        return (True, cleaned_bboxes, class_ids)

    except Exception as e:
        logger.error(f"Lỗi đọc file nhãn {label_path}: {e}")
        return (False, [], [])


def scan_and_clean_dataset(
    raw_dir: str,
    allow_empty: bool = False
) -> List[Tuple[str, str, List[List[float]], List[int]]]:
    """
    Quét toàn bộ thư mục raw, loại bỏ file mồ côi (ảnh không nhãn hoặc nhãn không ảnh)
    và loại bỏ ảnh hỏng.
    
    Returns:
        Danh sách các mẫu hợp lệ: [(img_path, lbl_path, bboxes, class_ids), ...]
    """
    logger.info(f"Bắt đầu quét và kiểm tra dữ liệu trong thư mục: {raw_dir}")
    
    # Tìm kiếm tất cả ảnh
    image_paths = []
    for root, _, files in os.walk(raw_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_IMAGE_EXTS:
                image_paths.append(os.path.join(root, f))

    logger.info(f"Tìm thấy tổng cộng {len(image_paths)} file ảnh ứng viên.")

    valid_samples = []
    corrupted_images = 0
    missing_labels = 0
    invalid_labels = 0

    for img_path in tqdm(image_paths, desc="Quét tính toàn vẹn"):
        # 1. Kiểm tra ảnh
        img_dims = validate_image_file(img_path)
        if img_dims is None:
            corrupted_images += 1
            continue

        # 2. Tìm file nhãn tương ứng (cùng tên, đổi đuôi thành .txt)
        base_name = os.path.splitext(img_path)[0]
        lbl_path = base_name + ".txt"

        # Nếu không có ở cùng thư mục, thử tìm ở thư mục song song 'labels' (cấu trúc YOLO thông dụng)
        if not os.path.exists(lbl_path):
            parent_dir = os.path.dirname(img_path)
            if os.path.basename(parent_dir) == "images":
                alt_lbl_path = os.path.join(os.path.dirname(parent_dir), "labels", os.path.basename(base_name) + ".txt")
                if os.path.exists(alt_lbl_path):
                    lbl_path = alt_lbl_path

        if not os.path.exists(lbl_path):
            missing_labels += 1
            continue

        # 3. Kiểm tra file nhãn
        is_valid, bboxes, class_ids = validate_and_clean_yolo_labels(lbl_path, allow_empty=allow_empty)
        if not is_valid:
            invalid_labels += 1
            continue

        valid_samples.append((img_path, lbl_path, bboxes, class_ids))

    logger.info("=" * 65)
    logger.info("KẾT QUẢ QUÉT TÍNH TOÀN VẸN BỘ DỮ LIỆU:")
    logger.info(f"  - Mẫu dữ liệu hợp lệ (sẵn sàng sử dụng) : {len(valid_samples)}")
    logger.info(f"  - Ảnh bị lỗi cấu trúc / hỏng (Corrupted) : {corrupted_images}")
    logger.info(f"  - Ảnh thiếu file nhãn .txt (Orphaned)   : {missing_labels}")
    logger.info(f"  - File nhãn sai cú pháp / rỗng          : {invalid_labels}")
    logger.info("=" * 65)

    return valid_samples


# ==============================================================================
# 2. CẤU HÌNH ALBUMENTATIONS PIPELINE (CHỐNG OVERFITTING & BIẾN THIÊN ÁNH SÁNG)
# ==============================================================================

def get_robust_augmentation_pipeline() -> Optional[Any]:
    """
    Xây dựng pipeline tăng cường dữ liệu chuyên sâu cho bài toán phát hiện hành vi hút thuốc
    trong bối cảnh camera giám sát an ninh (CCTV):
    - Đổi màu, tương phản: Mô phỏng ánh sáng ngày/đêm/đèn neon
    - Mờ nhòe: Mô phỏng người đi lại, camera rung lắc
    - Biến dạng phối cảnh: Mô phỏng góc quay trên cao nghiêng 45-60 độ
    - Che khuất: Mô phỏng ngón tay hoặc vật dụng che khuất điếu thuốc
    """
    if not ALBUMENTATIONS_AVAILABLE:
        logger.warning("Thư viện 'albumentations' chưa được cài đặt. Bỏ qua bước Albumentations augmentations.")
        return None

    return A.Compose([
        # 1. Biến thiên ánh sáng và màu sắc
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.8),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=25, p=0.8),
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),
        ], p=0.85),

        # 2. Làm mờ mô phỏng chuyển động / camera mất nét
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=0.5),
            A.GaussianBlur(blur_limit=5, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
        ], p=0.4),

        # 3. Phối cảnh góc nghiêng & xoay nhẹ (Camera CCTV gắn trên cao)
        A.Perspective(scale=(0.04, 0.08), keep_size=True, p=0.4),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=12, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.HorizontalFlip(p=0.5),

        # 4. Che khuất một phần (Occlusion - ngón tay, cốc nước che điếu thuốc)
        A.CoarseDropout(
            max_holes=6,
            max_height=32,
            max_width=32,
            min_holes=1,
            min_height=12,
            min_width=12,
            fill_value=0,
            p=0.3
        )
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=16, min_visibility=0.25))


def apply_augmentations(
    image_bgr: np.ndarray,
    bboxes: List[List[float]],
    class_ids: List[int],
    pipeline: Any,
    num_variants: int = 1
) -> List[Tuple[np.ndarray, List[List[float]], List[int]]]:
    """Sinh ra các phiên bản ảnh tăng cường kèm nhãn bounding box đã được ánh xạ chính xác."""
    if pipeline is None:
        return []

    augmented_results = []
    # Albumentations yêu cầu ảnh định dạng RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    for _ in range(num_variants):
        try:
            res = pipeline(image=image_rgb, bboxes=bboxes, class_labels=class_ids)
            aug_bgr = cv2.cvtColor(res['image'], cv2.COLOR_RGB2BGR)
            aug_bboxes = [list(box) for box in res['bboxes']]
            aug_labels = list(res['class_labels'])

            # Chỉ giữ lại nếu còn ít nhất 1 bounding box hợp lệ sau khi augment
            if aug_bboxes:
                augmented_results.append((aug_bgr, aug_bboxes, aug_labels))
        except Exception as e:
            logger.debug(f"Bỏ qua 1 phép augment lỗi: {e}")
            continue

    return augmented_results


# ==============================================================================
# 3. CHIA TẬP TRAIN/VAL VÀ LƯU TRỮ CHUẨN THƯ MỤC
# ==============================================================================

def write_yolo_label_file(file_path: str, bboxes: List[List[float]], class_ids: List[int]):
    """Ghi nhãn ra file .txt theo định dạng chuẩn YOLO: cls_id xc yc w h"""
    with open(file_path, "w", encoding="utf-8") as f:
        for cls_id, box in zip(class_ids, bboxes):
            f.write(f"{cls_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")


def split_and_organize_dataset(
    samples: List[Tuple[str, str, List[List[float]], List[int]]],
    output_base_dir: str = "datasets",
    train_ratio: float = 0.8,
    seed: int = 42,
    enable_aug: bool = False,
    num_aug_per_train_image: int = 1
):
    """
    Chia các mẫu thành train và val theo tỷ lệ chỉ định, di chuyển/copy file
    vào đúng cây thư mục chuẩn YOLO và thực hiện Data Augmentation nếu có yêu cầu.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Xáo trộn danh sách
    indices = list(range(len(samples)))
    random.shuffle(indices)

    split_idx = int(len(samples) * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    # Tạo các thư mục đích
    train_img_dir = os.path.join(output_base_dir, "train", "images")
    train_lbl_dir = os.path.join(output_base_dir, "train", "labels")
    val_img_dir = os.path.join(output_base_dir, "val", "images")
    val_lbl_dir = os.path.join(output_base_dir, "val", "labels")

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    aug_pipeline = get_robust_augmentation_pipeline() if enable_aug else None

    # Thống kê phân bố lớp
    train_cls_stats: Dict[int, int] = {}
    val_cls_stats: Dict[int, int] = {}

    logger.info(f"Đang tổ chức dữ liệu: Tập Train ({len(train_indices)} mẫu), Tập Val ({len(val_indices)} mẫu)...")

    # 1. Xử lý tập Train
    for idx in tqdm(train_indices, desc="Ghi tập Train"):
        img_path, lbl_path, bboxes, class_ids = samples[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1]

        # Sao chép file gốc
        dst_img = os.path.join(train_img_dir, f"{base_name}{ext}")
        dst_lbl = os.path.join(train_lbl_dir, f"{base_name}.txt")
        shutil.copy2(img_path, dst_img)
        write_yolo_label_file(dst_lbl, bboxes, class_ids)

        for c in class_ids:
            train_cls_stats[c] = train_cls_stats.get(c, 0) + 1

        # Thực hiện Data Augmentation trên tập Train nếu được bật
        if enable_aug and aug_pipeline:
            img_bgr = cv2.imread(img_path)
            if img_bgr is not None:
                aug_samples = apply_augmentations(img_bgr, bboxes, class_ids, aug_pipeline, num_variants=num_aug_per_train_image)
                for aug_i, (aug_img, aug_boxes, aug_lbls) in enumerate(aug_samples, 1):
                    aug_base = f"{base_name}_aug{aug_i}"
                    cv2.imwrite(os.path.join(train_img_dir, f"{aug_base}.jpg"), aug_img)
                    write_yolo_label_file(os.path.join(train_lbl_dir, f"{aug_base}.txt"), aug_boxes, aug_lbls)
                    for c in aug_lbls:
                        train_cls_stats[c] = train_cls_stats.get(c, 0) + 1

    # 2. Xử lý tập Val (KHÔNG augment tập Val để đảm bảo tính khách quan khi đánh giá)
    for idx in tqdm(val_indices, desc="Ghi tập Val"):
        img_path, lbl_path, bboxes, class_ids = samples[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1]

        dst_img = os.path.join(val_img_dir, f"{base_name}{ext}")
        dst_lbl = os.path.join(val_lbl_dir, f"{base_name}.txt")
        shutil.copy2(img_path, dst_img)
        write_yolo_label_file(dst_lbl, bboxes, class_ids)

        for c in class_ids:
            val_cls_stats[c] = val_cls_stats.get(c, 0) + 1

    # Báo cáo tổng kết
    print("\n" + "=" * 65)
    print(" BÁO CÁO THỐNG KÊ BỘ DỮ LIỆU SAU TIỀN XỬ LÝ")
    print("=" * 65)
    total_train_imgs = len(os.listdir(train_img_dir))
    total_val_imgs = len(os.listdir(val_img_dir))
    print(f"* Tổng số ảnh Train : {total_train_imgs} (bao gồm cả ảnh tăng cường nếu bật)")
    print(f"* Tổng số ảnh Val   : {total_val_imgs}")
    print("\n* Phân bố Bounding Box theo Class ID trong tập Train:")
    for cid in sorted(train_cls_stats.keys()):
        print(f"    - Class {cid}: {train_cls_stats[cid]} boxes")
    print("\n* Phân bố Bounding Box theo Class ID trong tập Val:")
    for cid in sorted(val_cls_stats.keys()):
        print(f"    - Class {cid}: {val_cls_stats[cid]} boxes")
    print("=" * 65)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Dataset Preprocessing & Augmentation Pipeline for Smoking Detection"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="datasets/raw",
        help="Đường dẫn thư mục chứa dữ liệu thô (ảnh và nhãn txt)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets",
        help="Thư mục gốc đích để lưu datasets/train và datasets/val."
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Tỷ lệ phân chia tập Train (mặc định 0.8 ứng với 80%% Train, 20%% Val)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed để tái lập kết quả phân chia tập dữ liệu."
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Bật các phép tăng cường dữ liệu Albumentations trên tập Train."
    )
    parser.add_argument(
        "--num-aug",
        type=int,
        default=1,
        help="Số lượng biến thể tăng cường sinh ra cho mỗi ảnh Train."
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Cho phép giữ lại ảnh không có vật thể (negative samples) để giảm False Positives."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if not os.path.exists(args.input_dir):
        logger.error(f"Thư mục đầu vào không tồn tại: {args.input_dir}")
        print(f"[GỢI Ý] Vui lòng đặt ảnh và file nhãn .txt vào '{args.input_dir}'")
        print("Hoặc chạy: python src/download_datasets.py để tải dữ liệu mẫu.")
        return

    # Quét và làm sạch dữ liệu
    valid_samples = scan_and_clean_dataset(args.input_dir, allow_empty=args.allow_empty)

    if not valid_samples:
        logger.warning(f"Không tìm thấy mẫu dữ liệu hợp lệ nào trong '{args.input_dir}'.")
        return

    # Phân chia train/val và tăng cường dữ liệu
    split_and_organize_dataset(
        samples=valid_samples,
        output_base_dir=args.output_dir,
        train_ratio=args.split_ratio,
        seed=args.seed,
        enable_aug=args.augment,
        num_aug_per_train_image=args.num_aug
    )
    logger.info("Hoàn tất tiền xử lý và chuẩn hóa dataset!")


if __name__ == "__main__":
    main()
