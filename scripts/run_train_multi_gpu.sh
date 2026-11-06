#!/bin/bash
# ==============================================================================
# SCRIPT KÍCH HOẠT HUẤN LUYỆN MULTI-GPU (2x NVIDIA RTX A4000)
# Tận dụng công nghệ Distributed Data Parallel (DDP) & FP16 Mixed Precision
# ==============================================================================

set -e

# Chỉ định 2 card GPU A4000
export CUDA_VISIBLE_DEVICES=0,1

# Tối ưu hóa NCCL cho multi-GPU DDP trên cùng 1 node
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4

echo "=================================================================="
echo " BẮT ĐẦU HUẤN LUYỆN MODEL TRÊN CỤM 2x NVIDIA RTX A4000"
echo " Thiết bị         : GPU 0 và GPU 1 (Tổng 32GB VRAM)"
echo " Mô hình khởi tạo : YOLOv8s (hoặc YOLOv11s)"
echo " Chế độ tối ưu    : FP16 Mixed Precision (amp=True), DDP"
echo " Batch size       : 32 (16 ảnh / GPU)"
echo " Workers          : 8 luồng async nạp dữ liệu"
echo "=================================================================="

# Lựa chọn mô hình: yolov8s.pt hoặc yolo11s.pt
MODEL_WEIGHTS=${1:-"yolov8s.pt"}
BATCH_SIZE=${2:-32}
EPOCHS=${3:-100}

python src/train_baseline.py \
    --model "$MODEL_WEIGHTS" \
    --data configs/smoking_dataset.yaml \
    --hyp configs/hyperparameters.yaml \
    --epochs "$EPOCHS" \
    --batch "$BATCH_SIZE" \
    --imgsz 640 \
    --device "0,1" \
    --workers 8 \
    --patience 20

echo -e "\n[HOÀN TẤT] Quá trình huấn luyện kết thúc!"
echo "Trọng số tối ưu nhất được lưu tại: runs/detect/smoking_baseline_2x_a4000/weights/best.pt"
