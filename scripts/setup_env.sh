#!/bin/bash
# ==============================================================================
# SCRIPT THIẾT LẬP MÔI TRƯỜNG DÀNH CHO SERVER 2x NVIDIA RTX A4000
# Hệ điều hành: Ubuntu 20.04 / 22.04 / Debian 11/12
# ==============================================================================

set -e

echo "=================================================================="
echo " KHỞI TẠO MÔI TRƯỜNG AI HUẤN LUYỆN: SMOKING DETECTION SYSTEM"
echo "=================================================================="

# 1. Kiểm tra driver NVIDIA và GPU
echo "[1/4] Kiểm tra thông tin GPU NVIDIA..."
if ! command -v nvidia-smi &> /dev/null
then
    echo "LỖI: Không tìm thấy nvidia-smi. Vui lòng cài đặt NVIDIA Driver trước!"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# 2. Tạo môi trường ảo Python Virtual Environment
ENV_NAME="smoking_env"
echo -e "\n[2/4] Tạo môi trường ảo Python '${ENV_NAME}'..."
if [ ! -d "$ENV_NAME" ]; then
    python3 -m venv $ENV_NAME
    echo "Đã tạo môi trường ảo thành công."
else
    echo "Môi trường ảo $ENV_NAME đã tồn tại."
fi

# Kích hoạt venv
source $ENV_NAME/bin/activate

# 3. Nâng cấp pip và cài đặt PyTorch với CUDA hỗ trợ Ampere (RTX A4000)
echo -e "\n[3/4] Cài đặt PyTorch với CUDA 12.1..."
pip install --upgrade pip wheel setuptools
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Cài đặt các gói phụ thuộc dự án
echo -e "\n[4/4] Cài đặt các thư viện trong requirements.txt..."
pip install -r requirements.txt

echo -e "\n=================================================================="
echo " HOÀN TẤT CÀI ĐẶT MÔI TRƯỜNG!"
echo " Kích hoạt môi trường bằng lệnh: source $ENV_NAME/bin/activate"
echo " Kiểm tra phần cứng bằng lệnh : python src/train_baseline.py --dry-run"
echo "=================================================================="
