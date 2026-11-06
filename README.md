# ÁP DỤNG CÔNG NGHỆ HỌC SÂU HỖ TRỢ CẢNH BÁO PHÁT HIỆN HÀNH VI HÚT THUỐC LÁ Ở NƠI CÔNG CỘNG

Hệ thống mã nguồn Baseline hoàn chỉnh và các công cụ tiện ích cho đề tài Nghiên cứu Khoa học (NCKH) phát hiện hành vi hút thuốc lá qua hệ thống camera giám sát an ninh (CCTV) nơi công cộng.

Mã nguồn được tối ưu hóa đặc thù cho máy chủ huấn luyện trang bị **2x NVIDIA RTX A4000 (16GB VRAM mỗi card, tổng 32GB VRAM)** với kiến trúc song song đa GPU (DDP - Distributed Data Parallel) và huấn luyện độ chính xác hỗn hợp FP16 (AMP).

---

## 1. CẤU TRÚC THƯ MỤC DỰ ÁN (PROJECT STRUCTURE)

```text
Smoking_Detect_Public/
├── configs/
│   ├── smoking_dataset.yaml      # Cấu hình dữ liệu YOLO (train/val paths, class names)
│   └── hyperparameters.yaml      # Siêu tham số nâng cao (Box loss, Mosaic, Mixup, Augmentations)
├── datasets/
│   ├── raw/                      # Thư mục lưu dữ liệu thô ban đầu sau khi tải về
│   ├── train/
│   │   ├── images/               # Ảnh tập huấn luyện
│   │   └── labels/               # Nhãn txt tương ứng [class_id xc yc w h]
│   └── val/
│       ├── images/               # Ảnh tập kiểm định
│       └── labels/               # Nhãn txt kiểm định
├── src/
│   ├── __init__.py               # Package initializer
│   ├── utils.py                  # Thuật toán không gian (Spatial Heuristics), HUD, Logger, IoU
│   ├── download_datasets.py      # Tự động hóa tìm kiếm và tải dữ liệu từ Roboflow Universe / Kaggle
│   ├── data_preprocessing.py     # Quét tính toàn vẹn, chia 80-20, Albumentations chống overfitting
│   ├── train_baseline.py         # Huấn luyện YOLOv8s/YOLOv11s tối ưu 2x RTX A4000 (DDP, FP16 AMP)
│   └── inference_demo.py         # Xử lý Video/Webcam/RTSP thời gian thực, lọc giả lập, cảnh báo đỏ
├── scripts/
│   ├── setup_env.sh              # Tự động cài đặt môi trường PyTorch CUDA 12.1 và dependencies
│   └── run_train_multi_gpu.sh    # Script kích hoạt train multi-GPU 1 click
├── violations/                   # Thư mục lưu vết vi phạm
│   ├── snapshots/                # Ảnh bằng chứng vi phạm có đóng watermark thời gian
│   └── violation_logs.csv        # Nhật ký vi phạm chi tiết (timestamp, camera_id, conf, bbox)
├── requirements.txt              # Danh sách thư viện Python cần thiết
└── README.md                     # Tài liệu hướng dẫn chi tiết
```

---

## 2. HƯỚNG DẪN THIẾT LẬP MÔI TRƯỜNG (SETUP ENVIRONMENT)

Chạy trên máy chủ Linux (Ubuntu 20.04/22.04 hoặc Debian 11/12):

```bash
# Cách 1: Thiết lập tự động thông qua script
chmod +x scripts/setup_env.sh scripts/run_train_multi_gpu.sh
./scripts/setup_env.sh

# Cách 2: Thiết lập thủ công qua terminal
python3 -m venv smoking_env
source smoking_env/bin/activate

# Cài đặt PyTorch với CUDA 12.1 (Tối ưu cho kiến trúc Ampere GA104 của RTX A4000)
pip install --upgrade pip wheel setuptools
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

---

## 3. THU THẬP & TẢI DỮ LIỆU CÔNG KHAI (DATASET ACQUISITION)

Module `src/download_datasets.py` tích hợp sẵn danh mục các bộ dữ liệu mã nguồn mở chất lượng cao nhất hiện nay, gán nhãn chuẩn YOLO:

| ID Bộ dữ liệu | Nguồn | Các nhãn chính | Quy mô | Đặc điểm |
| :--- | :--- | :--- | :--- | :--- |
| `roboflow_smoking_behavior` | Roboflow 100 | `smoking`, `not_smoking` | ~2,500 ảnh | Benchmark chuẩn, tập trung vào hành vi |
| `roboflow_cigarette_v8` | Roboflow | `cigarette`, `smoke`, `person_smoking` | ~3,800 ảnh | Chi tiết điếu thuốc và khói |
| `roboflow_smoking_calling` | Roboflow | `smoking`, `calling`, `normal` | ~4,200 ảnh | Phân biệt hút thuốc vs. nghe điện thoại |
| `kaggle_smoking_detection` | Kaggle | `smoking`, `cigarette` | ~5,000 ảnh | Thu thập từ CCTV công cộng thực tế |

### Câu lệnh sử dụng:

```bash
# 1. Xem danh mục chi tiết các bộ dữ liệu
python src/download_datasets.py --list

# 2. Tải trực tiếp từ Roboflow Universe (cung cấp API Key miễn phí từ roboflow.com)
python src/download_datasets.py \
    --source roboflow \
    --dataset-id roboflow_smoking_behavior \
    --api-key YOUR_ROBOFLOW_API_KEY \
    --output-dir datasets/raw

# 3. Tải từ Kaggle (yêu cầu file ~/.kaggle/kaggle.json)
python src/download_datasets.py \
    --source kaggle \
    --dataset-id kaggle_smoking_detection \
    --output-dir datasets/raw
```

---

## 4. TIỀN XỬ LÝ & TĂNG CƯỜNG DỮ LIỆU (PREPROCESSING & AUGMENTATION)

Module `src/data_preprocessing.py` thực hiện:
1. **Kiểm tra tính toàn vẹn**: Phát hiện và loại bỏ ảnh bị hỏng cấu trúc (corrupt images), loại bỏ các file nhãn mồ côi (không có ảnh) hoặc ảnh không nhãn.
2. **Chuẩn hóa YOLO Bounding Box**: Kiểm tra cú pháp, dung sai tọa độ $[x_c, y_c, w, h] \in [0.0, 1.0]$.
3. **Phân chia 80/20 Train/Val**: Cố định `seed=42` đảm bảo tính tái lập kết quả nghiên cứu khoa học.
4. **Data Augmentation với Albumentations**:
   - `RandomBrightnessContrast`, `HueSaturationValue`, `CLAHE`: Chống overfitting trong điều kiện ánh sáng thay đổi (ngày, đêm, ngược sáng, đèn huỳnh quang công cộng).
   - `MotionBlur`, `GaussianBlur`: Giả lập người chuyển động và camera CCTV rung lắc.
   - `Perspective`, `ShiftScaleRotate`: Giả lập góc nhìn camera giám sát nghiêng trên cao ($45^\circ - 60^\circ$).
   - `CoarseDropout`: Giả lập tình huống ngón tay hoặc vật cản che khuất một phần điếu thuốc.

```bash
# Chạy tiền xử lý dữ liệu và kích hoạt tăng cường dữ liệu:
python src/data_preprocessing.py \
    --input-dir datasets/raw \
    --output-dir datasets \
    --split-ratio 0.8 \
    --seed 42 \
    --augment \
    --num-aug 1
```

---

## 5. TỐI ƯU HÓA HUẤN LUYỆN CHO CỤM 2x NVIDIA RTX A4000

### 5.1. Bảng Thông số Tối ưu Phần cứng:

| Tham số cấu hình | Giá trị thiết lập | Cơ sở tối ưu hóa kỹ thuật |
| :--- | :--- | :--- |
| **Thiết bị (Device)** | `device='0,1'` | Kích hoạt PyTorch DDP (Distributed Data Parallel) song song trên 2 GPU độc lập. |
| **Độ chính xác tính toán** | `amp=True` | Kích hoạt Automatic Mixed Precision (FP16), tận dụng 192 Tensor Cores thế hệ 3 của RTX A4000, tăng tốc ~2.5x và giảm 50% dung lượng VRAM. |
| **Batch Size** | `batch=32` (hoặc `64`) | Khai thác hiệu quả 16GB VRAM mỗi GPU ($16 \times 2 = 32$), giữ gradient ổn định. |
| **Số luồng CPU (Workers)** | `workers=8` | 4 workers mỗi GPU chạy song song nạp batch và augment, loại bỏ nghẽn cổ chai I/O. |
| **Kích thước ảnh (`imgsz`)** | `640` (hoặc `1280`) | $640 \times 640$ cho baseline cân bằng, hoặc $1280 \times 1280$ cho camera góc rộng. |
| **Bộ tối ưu (Optimizer)** | `AdamW` | Kết hợp weight decay ($0.0005$) và Cosine Annealing learning rate schedule ($lr_0=0.001$). |
| **Early Stopping** | `patience=20` | Tự động ngắt khi $mAP_{50-95}$ không tăng sau 20 epoch để chống hiện tượng quá khớp. |

### 5.2. Câu lệnh chạy huấn luyện:

#### Cách 1: Chạy trực tiếp qua Python script:
```bash
# Kiểm tra phần cứng trước khi chạy (Dry Run - không tốn tài nguyên):
python src/train_baseline.py --dry-run

# Huấn luyện baseline với YOLOv8s trên 2x RTX A4000:
python src/train_baseline.py \
    --model yolov8s.pt \
    --data configs/smoking_dataset.yaml \
    --hyp configs/hyperparameters.yaml \
    --epochs 100 \
    --batch 32 \
    --device "0,1" \
    --workers 8 \
    --patience 20

# Huấn luyện với kiến trúc mới nhất YOLOv11s:
python src/train_baseline.py \
    --model yolo11s.pt \
    --data configs/smoking_dataset.yaml \
    --epochs 100 \
    --batch 32 \
    --device "0,1" \
    --workers 8
```

#### Cách 2: Chạy 1 click bằng Bash Script:
```bash
# Cú pháp: ./scripts/run_train_multi_gpu.sh <model_name> <batch_size> <epochs>
./scripts/run_train_multi_gpu.sh yolov8s.pt 32 100
```

---

## 6. HỆ THỐNG SUY LUẬN & CẢNH BÁO THỜI GIAN THỰC (REAL-TIME INFERENCE)

### 6.1. Cơ chế Lọc Không Gian & Thời Gian (Spatial-Temporal Verification):
1. **Lọc Không Gian (Spatial Check)**: Khi mô hình phát hiện điếu thuốc (`cigarette`), hệ thống kiểm tra khoảng cách hình học tương đối giữa điếu thuốc và vùng thân trên/mặt của người (`person`). Nếu điếu thuốc nằm cô lập dưới đất hoặc cách xa người, hệ thống tự động loại bỏ để triệt tiêu các báo động giả (False Positives) do bút bi, que tăm, ống hút.
2. **Lọc Thời Gian (Temporal Smoother)**: Áp dụng Moving Window (cần ít nhất 3 trong 8 frames liên tiếp xác nhận vi phạm) nhằm loại bỏ hiện tượng nhấp nháy cảnh báo (flickering).
3. **Giao diện HUD Giám Sát**:
   - Khung viền Bounding Box màu đỏ bo góc nổi bật.
   - Banner màu đỏ rực rỡ ở phía trên: `!! CANH BAO: HUT THUOC LA !!` kèm đồng hồ FPS, mã camera và timestamp.
4. **Nhật ký & Bằng chứng**:
   - Ghi thông tin chi tiết vào `violations/violation_logs.csv`.
   - Tự động đóng dấu watermark thời gian và chụp ảnh bằng chứng lưu vào `violations/snapshots/`.
   - Có cơ chế Cooldown ($3$ giây) để tránh ghi trùng lặp liên tục cho cùng một đối tượng.

### 6.2. Câu lệnh chạy suy luận:

```bash
# 1. Chạy với Webcam máy tính / Laptop:
python src/inference_demo.py --source 0 --conf 0.45

# 2. Chạy với file video demo (có lưu video kết quả):
python src/inference_demo.py \
    --source data/test_cctv.mp4 \
    --weights runs/detect/smoking_baseline_2x_a4000/weights/best.pt \
    --conf 0.45 \
    --output runs/detect/alert_result.mp4

# 3. Chạy với luồng camera giám sát an ninh (RTSP Stream) ở chế độ Server Headless:
python src/inference_demo.py \
    --source "rtsp://admin:password123@192.168.1.100:554/stream1" \
    --weights runs/detect/smoking_baseline_2x_a4000/weights/best.pt \
    --conf 0.45 \
    --camera-id "CAM_CONG_VIEN_01" \
    --no-show
```

---

## 7. CẤU TRÚC LOGGING VI PHẠM (VIOLATION AUDIT TRAIL)

Mỗi sự việc vi phạm được ghi tự động vào file `violations/violation_logs.csv` theo định dạng:

```csv
Timestamp,Camera_ID,Confidence,BBox_X1,BBox_Y1,BBox_X2,BBox_Y2,Event_Type,Snapshot_File
2026-11-06 14:22:15,CAM_CONG_VIEN_01,0.8742,420,185,580,490,SMOKING_BEHAVIOR,violation_CAM_CONG_VIEN_01_20261106_142215.jpg
```

---

## 8. BẢNG SO SÁNH CÁC MÔ HÌNH BASELINE PHÙ HỢP CHO ĐỀ TÀI

| Mô hình | Số lượng tham số (Params) | FLOPs (640x640) | Ưu điểm trong NCKH | Thời gian huấn luyện (100 Epochs, 2x A4000) |
| :--- | :--- | :--- | :--- | :--- |
| **YOLOv8s** | 11.2M | 28.6G | Cộng đồng lớn, độ ổn định cao, là mô hình đối sánh (benchmark baseline) chuẩn mực. | ~1.5 - 2.0 giờ |
| **YOLOv8m** | 25.9M | 78.9G | Cải thiện độ chính xác trên vật thể nhỏ (điếu thuốc ở khoảng cách xa). | ~3.0 - 3.5 giờ |
| **YOLOv11s** | 9.4M | 21.5G | Kiến trúc C3k2 và Attention mới, tốc độ suy luận nhanh hơn, tiết kiệm tài nguyên. | ~1.2 - 1.8 giờ |

---

## 9. GIẤY PHÉP & BẢN QUYỀN (LICENSE)

Dự án được xây dựng phục vụ mục đích Nghiên cứu Khoa học và Ứng dụng Trí tuệ Nhân tạo trong giám sát an ninh trật tự công cộng.
