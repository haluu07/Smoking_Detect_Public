#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Module: train_baseline.py
Mục đích: Huấn luyện mô hình phát hiện hành vi hút thuốc lá (YOLOv8s / YOLOv11s)
          được TỐI ƯU HÓA TUYỆT ĐỐI cho hệ thống 2x NVIDIA RTX A4000 (16GB VRAM mỗi card).
          
Đặc tả kỹ thuật tối ưu hóa phần cứng (Hardware Acceleration Profile):
  - Multi-GPU Architecture: Distributed Data Parallel (DDP) trên 2 GPUs (device='0,1')
  - Precision: Automatic Mixed Precision (amp=True) kích hoạt Tensor Cores thế hệ 3
  - Batch Sizing: Batch 32 hoặc 64 (16-32 per GPU) tối ưu độ phủ VRAM 16GB
  - I/O Throughput: workers=8 (4 worker/GPU) kết hợp pin_memory=True chống nghẽn I/O
  - Convergence: AdamW optimizer + Cosine Annealing LR Schedule + Early Stopping
================================================================================
"""

import os
import sys
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("TrainBaseline")


def check_hardware_environment() -> Dict[str, Any]:
    """
    Kiểm tra thông số phần cứng của server: Số GPU, Tên GPU, VRAM, Phiên bản CUDA và CPU.
    """
    print("\n" + "=" * 75)
    print(" BẢNG KIỂM TRA PHẦN CỨNG SERVER (HARDWARE VERIFICATION)")
    print("=" * 75)
    
    if not TORCH_AVAILABLE:
        print("* LƯU Ý: Thư viện 'torch' chưa được cài đặt trong môi trường hiện tại.")
        print("  Vui lòng chạy: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("=" * 75 + "\n")
        return {"cuda_available": False, "gpu_count": 0, "cpu_cores": 1, "ram_gb": 0.0}

    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    
    print(f"* PyTorch Version  : {torch.__version__}")
    print(f"* CUDA Available   : {cuda_available}")
    if cuda_available:
        print(f"* CUDA Version     : {torch.version.cuda}")
        print(f"* Số lượng GPU     : {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            total_vram_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            print(f"    - GPU {i}: {gpu_name} | VRAM: {total_vram_gb:.2f} GB")
    else:
        print("* CẢNH BÁO: Không phát hiện GPU CUDA! Quá trình huấn luyện trên CPU sẽ rất chậm.")

    cpu_cores = psutil.cpu_count(logical=True) if PSUTIL_AVAILABLE else os.cpu_count() or 1
    ram_gb = (psutil.virtual_memory().total / (1024 ** 3)) if PSUTIL_AVAILABLE else 0.0
    print(f"* CPU Cores        : {cpu_cores} threads")
    print(f"* System RAM       : {ram_gb:.2f} GB")
    print("=" * 75 + "\n")

    return {
        "cuda_available": cuda_available,
        "gpu_count": gpu_count,
        "cpu_cores": cpu_cores,
        "ram_gb": ram_gb
    }


def verify_dataset_yaml(yaml_path: str) -> str:
    """
    Kiểm tra sự tồn tại của file YAML và tính sẵn sàng của thư mục dữ liệu train/val.
    Trả về đường dẫn tuyệt đối của file cấu hình YAML.
    """
    abs_yaml = os.path.abspath(yaml_path)
    if not os.path.exists(abs_yaml):
        logger.error(f"Không tìm thấy file cấu hình dataset: {abs_yaml}")
        return ""

    import yaml
    try:
        with open(abs_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        base_path = cfg.get("path", "")
        # Nếu path là tương đối, tính từ thư mục chứa file yaml hoặc project root
        if not os.path.isabs(base_path):
            project_root = os.path.dirname(os.path.dirname(abs_yaml))
            resolved_base = os.path.join(project_root, base_path)
        else:
            resolved_base = base_path

        train_imgs = os.path.join(resolved_base, cfg.get("train", "train/images"))
        val_imgs = os.path.join(resolved_base, cfg.get("val", "val/images"))

        num_train = len([f for f in os.listdir(train_imgs) if not f.startswith(".")]) if os.path.exists(train_imgs) else 0
        num_val = len([f for f in os.listdir(val_imgs) if not f.startswith(".")]) if os.path.exists(val_imgs) else 0

        logger.info(f"Đã xác thực dữ liệu: Tập Train ({num_train} ảnh) | Tập Val ({num_val} ảnh)")
        if num_train == 0 or num_val == 0:
            logger.warning("[CHÚ Ý] Tập dữ liệu Train hoặc Val đang rỗng!")
            logger.warning("Vui lòng chạy tiền xử lý trước: python src/data_preprocessing.py --input-dir datasets/raw --output-dir datasets")

    except Exception as e:
        logger.debug(f"Bỏ qua bước đọc chi tiết yaml: {e}")

    return abs_yaml


def run_training(
    model_name: str = "yolov8s.pt",
    data_yaml: str = "configs/smoking_dataset.yaml",
    hyp_yaml: str = "configs/hyperparameters.yaml",
    epochs: int = 100,
    batch_size: int = 32,
    imgsz: int = 640,
    device: str = "0,1",
    workers: int = 8,
    project_dir: str = "runs/detect",
    run_name: str = "smoking_baseline_2x_a4000",
    patience: int = 20,
    cache_mode: str = "False",
    dry_run: bool = False
):
    """
    Kích hoạt tiến trình huấn luyện mô hình YOLO với các thiết lập tối ưu cho 2x RTX A4000.
    """
    # 1. Kiểm tra phần cứng
    hw_info = check_hardware_environment()

    # 2. Kiểm tra file cấu hình
    abs_data_yaml = verify_dataset_yaml(data_yaml)
    if not abs_data_yaml:
        sys.exit(1)
    data_yaml = abs_data_yaml

    # Kiểm tra cấu hình device
    if hw_info["gpu_count"] < 2 and ("0,1" in device or len(device.split(",")) > 1):
        logger.warning(f"Chỉ phát hiện {hw_info['gpu_count']} GPU, nhưng cấu hình yêu cầu '{device}'.")
        if hw_info["gpu_count"] == 1:
            logger.warning("Tự động chuyển về 1 GPU: device='0'")
            device = "0"
        elif hw_info["gpu_count"] == 0:
            logger.warning("Tự động chuyển về CPU: device='cpu'")
            device = "cpu"

    logger.info("=" * 75)
    logger.info(" CẤU HÌNH HUẤN LUYỆN TỐI ƯU CHO 2x NVIDIA RTX A4000:")
    logger.info(f"  - Model Architecture  : {model_name}")
    logger.info(f"  - Dataset Config      : {data_yaml}")
    logger.info(f"  - Epochs              : {epochs} (Early Stopping Patience: {patience})")
    logger.info(f"  - Batch Size          : {batch_size} (Chia đều trên {device})")
    logger.info(f"  - Image Resolution    : {imgsz}x{imgsz}")
    logger.info(f"  - Hardware Devices    : {device} (Distributed Data Parallel - DDP)")
    logger.info(f"  - Mixed Precision     : amp=True (FP16 Tensor Cores)")
    logger.info(f"  - DataLoader Workers  : {workers} (Parallel async prefetching)")
    logger.info(f"  - Project Output      : {os.path.join(project_dir, run_name)}")
    logger.info("=" * 75)

    if dry_run:
        logger.info("[DRY RUN] Đã kiểm tra hoàn tất toàn bộ cấu hình huấn luyện cho 2x RTX A4000.")
        logger.info("[DRY RUN] Khi chạy trên server có GPU và thư viện, lệnh sẽ kích hoạt DDP multi-GPU.")
        return

    # 3. Import Ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Thư viện 'ultralytics' chưa được cài đặt. Vui lòng chạy: pip install ultralytics")
        sys.exit(1)

    # 4. Khởi tạo mô hình
    logger.info(f"Đang tải trọng số tiền huấn luyện (pretrained weights): {model_name}...")
    model = YOLO(model_name)

    # Chuyển đổi cache mode
    parsed_cache = False
    if cache_mode.lower() in ("ram", "true"):
        parsed_cache = "ram" if cache_mode.lower() == "ram" else True

    # Xử lý tham số device linh hoạt cho Ultralytics
    parsed_device = device
    if isinstance(device, str):
        dev_clean = device.strip().strip("'\"")
        if dev_clean.isdigit():
            parsed_device = int(dev_clean)
        elif "," in dev_clean:
            parsed_device = [int(d.strip()) for d in dev_clean.split(",") if d.strip().isdigit()]
        else:
            parsed_device = dev_clean

    # 5. Khởi động huấn luyện
    # Toàn bộ tham số được tối ưu để khai thác tối đa phần cứng
    train_args = {
        "data": data_yaml,
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": imgsz,
        "device": parsed_device,
        "workers": workers,
        "amp": True,                 # Kích hoạt FP16 Mixed Precision
        "optimizer": "AdamW",        # Bộ tối ưu AdamW ổn định cho Computer Vision hiện đại
        "lr0": 0.001,                # Learning rate khởi tạo phù hợp với AdamW
        "lrf": 0.01,                 # Cosine decay xuống 1%
        "weight_decay": 0.0005,      # L2 Regularization chống overfitting
        "patience": patience,        # Dừng sớm nếu không tiến bộ
        "save": True,                # Lưu checkpoints
        "save_period": 10,           # Lưu checkpoint mỗi 10 epoch
        "cache": parsed_cache,       # Cache dữ liệu vào RAM nếu hệ thống đủ RAM
        "project": project_dir,
        "name": run_name,
        "exist_ok": True,
        "pretrained": True,
        "verbose": True,
        "seed": 42,
        "deterministic": True,       # Đảm bảo tính tái lập kết quả nghiên cứu khoa học
        "plots": True                # Tự động vẽ biểu đồ PR-curve, F1-curve, Confusion Matrix
    }

    # Nếu có file hyperparameters mở rộng, nạp thêm các tham số augmentation
    if os.path.exists(hyp_yaml):
        logger.info(f"Áp dụng cấu hình siêu tham số nâng cao từ: {hyp_yaml}")
        import yaml
        with open(hyp_yaml, "r", encoding="utf-8") as f:
            hyp_dict = yaml.safe_load(f)
            if isinstance(hyp_dict, dict):
                # Chỉ cập nhật các khóa liên quan đến augmentation và loss gain
                for k in ["box", "cls", "dfl", "mosaic", "mixup", "copy_paste", "fliplr", "hsv_h", "hsv_s", "hsv_v"]:
                    if k in hyp_dict:
                        train_args[k] = hyp_dict[k]

    logger.info("BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN MÔ HÌNH...")
    results = model.train(**train_args)

    logger.info("[HOÀN TẤT] Huấn luyện thành công!")
    save_dir = str(getattr(results, "save_dir", os.path.join(project_dir, run_name)))
    best_weight_path = os.path.join(save_dir, "weights", "best.pt")
    logger.info(f"Trọng số tối ưu nhất được lưu tại: {best_weight_path}")

    # Đánh giá trên tập val và lưu đúng thư mục project hiện tại
    logger.info("Đang tính toán ma trận đánh giá (Validation Metrics)...")
    val_results = model.val(project=save_dir, name="val_metrics", exist_ok=True)
    print("\n" + "=" * 75)
    print(" KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH BASELINE (VAL SET):")
    print(f"  - mAP50        : {val_results.box.map50:.4f}")
    print(f"  - mAP50-95     : {val_results.box.map:.4f}")
    print(f"  - Mean Precision : {val_results.box.mp:.4f}")
    print(f"  - Mean Recall    : {val_results.box.mr:.4f}")
    print("=" * 75)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình YOLOv8/v11 Phát hiện Hút thuốc lá - Tối ưu 2x NVIDIA RTX A4000"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="Mô hình khởi tạo (ví dụ: yolov8s.pt, yolov8m.pt, yolo11s.pt, yolo11m.pt)."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="configs/smoking_dataset.yaml",
        help="Đường dẫn file cấu hình dữ liệu dataset.yaml."
    )
    parser.add_argument(
        "--hyp",
        type=str,
        default="configs/hyperparameters.yaml",
        help="Đường dẫn file cấu hình siêu tham số hyperparameters.yaml."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Tổng số chu kỳ huấn luyện (epochs)."
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size tổng (khuyến nghị 32 hoặc 64 cho cụm 2x RTX A4000 16GB)."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Kích thước ảnh đầu vào (640 hoặc 1280 cho camera góc rộng CCTV)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0,1",
        help="Thiết bị tính toán GPU (mặc định '0,1' cho cụm 2x GPU)."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Số lượng luồng CPU nạp dữ liệu (khuyến nghị 8 hoặc 16)."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (dừng nếu sau N epochs không cải thiện mAP)."
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="False",
        choices=["False", "ram", "disk"],
        help="Lưu cache ảnh vào RAM hoặc disk để tăng tốc đọc dữ liệu."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ kiểm tra cấu hình và phần cứng, KHÔNG chạy huấn luyện thực tế."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_training(
        model_name=args.model,
        data_yaml=args.data,
        hyp_yaml=args.hyp,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        cache_mode=args.cache,
        dry_run=args.dry_run
    )
