#!/usr/bin/env bash
# ==============================================================================
# HỆ THỐNG GIÁM SÁT HÚT THUỐC LÁ NƠI CÔNG CỘNG
# Script tự động khởi chạy Web Demo & tạo đường dẫn Cloudflare Tunnel để Thuyết trình
# ==============================================================================

set -e

PORT=7860
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "==========================================================================="
echo " KHỞI CHẠY HỆ THỐNG DEMO PHỤC VỤ THUYẾT TRÌNH (CLOUDFLARE TUNNEL)"
echo "==========================================================================="

# 1. Kiểm tra gói phụ thuộc
echo "[1/4] Kiểm tra các thư viện Web API cần thiết..."
python3 -c "import fastapi, uvicorn, multipart" 2>/dev/null || {
    echo ">> Đang cài đặt FastAPI, Uvicorn, Python-Multipart..."
    pip install fastapi uvicorn python-multipart
}

# 2. Kiểm tra file trọng số mô hình
echo "[2/4] Kiểm tra trọng số mô hình tối ưu..."
if [ -f "runs/detect/smoking_baseline_2x_a4000/weights/best.pt" ]; then
    echo ">> Đã tìm thấy trọng số fine-tuned tại runs/detect/smoking_baseline_2x_a4000/weights/best.pt"
elif [ -f "/home/qtech/CheatingDetection/runs/detect/runs/detect/smoking_baseline_2x_a4000/weights/best.pt" ]; then
    echo ">> Phát hiện trọng số tại CheatingDetection, đang copy về thư mục hiện tại..."
    mkdir -p runs/detect
    cp -r /home/qtech/CheatingDetection/runs/detect/runs/detect/smoking_baseline_2x_a4000 runs/detect/
    echo ">> Copy thành công!"
fi

# 3. Kiểm tra công cụ Cloudflare Tunnel (cloudflared)
echo "[3/4] Kiểm tra công cụ cloudflared..."
if ! command -v cloudflared &> /dev/null; then
    echo "==========================================================================="
    echo " [LƯU Ý] 'cloudflared' chưa được cài trên server này."
    echo " Để tạo link public https cho ban giám khảo xem, anh chạy lệnh sau (1 lần duy nhất):"
    echo "   wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb"
    echo "   sudo dpkg -i cloudflared-linux-amd64.deb"
    echo "==========================================================================="
fi

# 4. Khởi chạy Web Server và Cloudflare Tunnel
echo "[4/4] Đang khởi chạy FastAPI Server tại cổng ${PORT}..."
echo "Local Dashboard: http://localhost:${PORT}"
echo "---------------------------------------------------------------------------"

if command -v cloudflared &> /dev/null; then
    echo ">> Đang tự động kết nối Cloudflare Tunnel..."
    # Chạy uvicorn background
    python3 app.py --port ${PORT} &
    SERVER_PID=$!
    sleep 2

    # Chạy cloudflared tunnel
    echo ">> Bật Cloudflare Tunnel công khai..."
    cloudflared tunnel --url http://localhost:${PORT}
    kill $SERVER_PID 2>/dev/null || true
else
    # Chạy bình thường
    echo ">> Đang chạy server nội bộ (Không có cloudflared). Chạy lệnh:"
    python3 app.py --port ${PORT}
fi
