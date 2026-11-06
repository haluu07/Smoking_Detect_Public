#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Module: download_datasets.py
Mục đích: Tự động hóa việc tìm kiếm, đề xuất danh mục và tải các bộ dữ liệu công khai
          (Open-source Datasets) chuẩn định dạng YOLO từ Roboflow Universe, Kaggle, 
          và GitHub phục vụ bài toán phát hiện hành vi hút thuốc lá.
================================================================================
"""

import os
import sys
import shutil
import zipfile
import tarfile
import argparse
import urllib.request
from typing import Dict, List, Any


# ==============================================================================
# DANH MỤC CÁC BỘ DỮ LIỆU CÔNG KHAI CHẤT LƯỢNG CAO (CURATED DATASET CATALOG)
# ==============================================================================
CURATED_DATASETS: Dict[str, Dict[str, Any]] = {
    # --------------------------------------------------------------------------
    # 1. ROBOFLOW UNIVERSE (Đã gán nhãn chuẩn YOLOv8/YOLOv11, chỉ cần tải về)
    # --------------------------------------------------------------------------
    "roboflow_smoking_behavior": {
        "source": "Roboflow Universe",
        "name": "Smoking Behavior Detection Dataset",
        "url": "https://universe.roboflow.com/roboflow-100/smoking-behavior-detection",
        "workspace": "roboflow-100",
        "project": "smoking-behavior-detection",
        "version": 1,
        "classes": ["smoking", "not_smoking"],
        "approx_images": 2500,
        "description": "Bộ dữ liệu tiêu chuẩn từ benchmark Roboflow-100, tập trung vào hành vi người hút thuốc.",
        "recommended": True
    },
    "roboflow_cigarette_v8": {
        "source": "Roboflow Universe",
        "name": "Cigarette and Smoke Detection",
        "url": "https://universe.roboflow.com/project-smoking/smoking-detection-yolo",
        "workspace": "project-smoking",
        "project": "smoking-detection-yolo",
        "version": 2,
        "classes": ["cigarette", "smoke", "person_smoking"],
        "approx_images": 3800,
        "description": "Gán nhãn chi tiết từng điếu thuốc, tàn khói và người hút thuốc trong nhà và ngoài trời.",
        "recommended": True
    },
    "roboflow_smoking_calling": {
        "source": "Roboflow Universe",
        "name": "Smoking and Calling In Public",
        "url": "https://universe.roboflow.com/security-cctv/smoking-calling-detection",
        "workspace": "security-cctv",
        "project": "smoking-calling-detection",
        "version": 1,
        "classes": ["smoking", "calling", "normal"],
        "approx_images": 4200,
        "description": "Đặc biệt hữu ích để phân biệt hành vi hút thuốc và nghe điện thoại (hạn chế False Positives).",
        "recommended": True
    },

    # --------------------------------------------------------------------------
    # 2. KAGGLE DATASETS (Bộ dữ liệu quy mô lớn, đa dạng bối cảnh)
    # --------------------------------------------------------------------------
    "kaggle_smoking_detection": {
        "source": "Kaggle",
        "name": "Smoking and Calling Detection (CCTV / Security)",
        "url": "https://www.kaggle.com/datasets/vitaminc/cigarette-smokie-dataset",
        "kaggle_slug": "vitaminc/cigarette-smokie-dataset",
        "classes": ["smoking", "cigarette"],
        "approx_images": 5000,
        "description": "Thu thập từ camera CCTV thực tế ở nơi công cộng, góc quay cao và điều kiện ánh sáng đa dạng.",
        "recommended": False
    },
    "kaggle_smoking_activity": {
        "source": "Kaggle",
        "name": "Smoking Activity Dataset for Edge Devices",
        "url": "https://www.kaggle.com/datasets/sujaykapadnis/smoking-detection-dataset",
        "kaggle_slug": "sujaykapadnis/smoking-detection-dataset",
        "classes": ["smoking", "not_smoking"],
        "approx_images": 3100,
        "description": "Phù hợp để huấn luyện mô hình nhẹ cho các thiết bị Edge AI hoặc baseline server.",
        "recommended": False
    }
}


def print_dataset_catalog():
    """In danh sách các bộ dữ liệu mã nguồn mở được đề xuất cho đề tài NCKH."""
    print("=" * 85)
    print(" DANH MỤC CÁC BỘ DỮ LIỆU CÔNG KHAI (OPEN-SOURCE DATASETS) CHO BÀI TOÁN HÚT THUỐC LÁ")
    print("=" * 85)
    for key, data in CURATED_DATASETS.items():
        rec_tag = " [KHUYẾN NGHỊ]" if data.get("recommended") else ""
        print(f"\n* ID: {key}{rec_tag}")
        print(f"  - Tên bộ dữ liệu : {data['name']}")
        print(f"  - Nguồn          : {data['source']}")
        print(f"  - Đường dẫn web  : {data['url']}")
        print(f"  - Các nhãn       : {', '.join(data['classes'])}")
        print(f"  - Quy mô xấp xỉ  : ~{data['approx_images']} ảnh")
        print(f"  - Đặc điểm       : {data['description']}")
    print("\n" + "=" * 85)


def download_from_roboflow(
    dataset_key: str,
    api_key: str,
    output_dir: str = "datasets/raw",
    model_format: str = "yolov8"
) -> bool:
    """
    Tải trực tiếp bộ dữ liệu từ Roboflow Universe thông qua Roboflow Python SDK.
    Dữ liệu tải về đã được phân sẵn train/val/test và định dạng bbox chuẩn YOLO.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[LỖI] Thư viện 'roboflow' chưa được cài đặt.")
        print("Vui lòng chạy: pip install roboflow")
        return False

    if dataset_key not in CURATED_DATASETS:
        print(f"[LỖI] ID bộ dữ liệu '{dataset_key}' không có trong danh mục.")
        return False

    meta = CURATED_DATASETS[dataset_key]
    if meta["source"] != "Roboflow Universe":
        print(f"[LỖI] '{dataset_key}' không phải từ Roboflow.")
        return False

    print(f"\n[INFO] Đang kết nối tới Roboflow Universe để tải: {meta['name']}...")
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(meta["workspace"]).project(meta["project"])
        version = project.version(meta["version"])
        
        # Tải dataset định dạng YOLO
        dataset = version.download(model_format=model_format, location=output_dir)
        print(f"[THÀNH CÔNG] Dữ liệu đã được tải về và lưu tại: {dataset.location}")
        return True
    except Exception as e:
        print(f"[LỖI] Quá trình tải từ Roboflow thất bại: {e}")
        print("[GỢI Ý] Hãy kiểm tra lại API Key hoặc đăng ký miễn phí tại https://roboflow.com")
        return False


def download_from_kaggle(
    dataset_key: str,
    output_dir: str = "datasets/raw"
) -> bool:
    """
    Tải bộ dữ liệu từ Kaggle thông qua Kaggle API CLI.
    Yêu cầu file cấu hình ~/.kaggle/kaggle.json đã được thiết lập.
    """
    if dataset_key not in CURATED_DATASETS:
        print(f"[LỖI] ID bộ dữ liệu '{dataset_key}' không có trong danh mục.")
        return False

    meta = CURATED_DATASETS[dataset_key]
    if meta["source"] != "Kaggle":
        print(f"[LỖI] '{dataset_key}' không phải từ Kaggle.")
        return False

    slug = meta["kaggle_slug"]
    os.makedirs(output_dir, exist_ok=True)
    target_zip = os.path.join(output_dir, f"{slug.replace('/', '_')}.zip")

    print(f"\n[INFO] Đang tải từ Kaggle: {slug}...")
    kaggle_cmd = f"kaggle datasets download -d {slug} -p {output_dir}"
    ret = os.system(kaggle_cmd)
    
    if ret != 0:
        print("[LỖI] Lệnh kaggle CLI thất bại. Vui lòng đảm bảo:")
        print("  1. Đã cài đặt kaggle: pip install kaggle")
        print("  2. Đã tải API Token 'kaggle.json' và đặt vào thư mục ~/.kaggle/kaggle.json")
        print("  3. Đã cấp quyền: chmod 600 ~/.kaggle/kaggle.json")
        return False

    # Tìm và giải nén file zip
    for fname in os.listdir(output_dir):
        if fname.endswith(".zip") and slug.split("/")[-1] in fname:
            zip_path = os.path.join(output_dir, fname)
            print(f"[INFO] Đang giải nén {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            print(f"[THÀNH CÔNG] Đã giải nén hoàn tất vào {output_dir}")
            return True

    return True


def prepare_sample_dataset(sample_dir: str = "datasets/sample_data", output_dir: str = "datasets/raw") -> bool:
    """
    Sử dụng bộ dữ liệu mẫu (Sample Dataset) được cung cấp sẵn trong kho GitHub
    để nạp vào pipeline kiểm thử ngay mà không cần kết nối internet.
    """
    if not os.path.exists(sample_dir):
        print(f"[LỖI] Thư mục dữ liệu mẫu không tồn tại: {sample_dir}")
        return False

    os.makedirs(output_dir, exist_ok=True)
    img_src = os.path.join(sample_dir, "images")
    lbl_src = os.path.join(sample_dir, "labels")

    count = 0
    if os.path.exists(img_src) and os.path.exists(lbl_src):
        for fname in os.listdir(img_src):
            shutil.copy2(os.path.join(img_src, fname), os.path.join(output_dir, fname))
            lbl_name = os.path.splitext(fname)[0] + ".txt"
            lbl_file = os.path.join(lbl_src, lbl_name)
            if os.path.exists(lbl_file):
                shutil.copy2(lbl_file, os.path.join(output_dir, lbl_name))
            count += 1
    else:
        # Nếu để chung trong 1 thư mục
        for fname in os.listdir(sample_dir):
            src_f = os.path.join(sample_dir, fname)
            if os.path.isfile(src_f):
                shutil.copy2(src_f, os.path.join(output_dir, fname))
                count += 1

    print(f"[THÀNH CÔNG] Đã sao chép {count} mẫu dữ liệu có sẵn từ '{sample_dir}' vào '{output_dir}'.")
    print("Bây giờ bạn có thể chạy: python src/data_preprocessing.py --input-dir datasets/raw --output-dir datasets")
    return True


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Dataset Downloader & Catalog Assistant cho đề tài Phát hiện Hút thuốc lá"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="In danh mục các bộ dữ liệu công khai chất lượng cao kèm đường dẫn."
    )
    parser.add_argument(
        "--prepare-sample",
        action="store_true",
        help="Sử dụng bộ dữ liệu mẫu (Sample Dataset) có sẵn trên repo để chạy thử nghiệm pipeline ngay."
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["roboflow", "kaggle"],
        help="Nền tảng tải dữ liệu (roboflow hoặc kaggle)."
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="roboflow_smoking_behavior",
        help="ID của bộ dữ liệu trong danh mục (ví dụ: roboflow_smoking_behavior, roboflow_cigarette_v8)."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("ROBOFLOW_API_KEY", ""),
        help="API Key của Roboflow (nếu không truyền sẽ đọc từ biến môi trường ROBOFLOW_API_KEY)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/raw",
        help="Thư mục đích để lưu trữ dữ liệu sau khi tải."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.prepare_sample:
        prepare_sample_dataset("datasets/sample_data", args.output_dir)
        return

    if args.list or len(sys.argv) == 1:
        print_dataset_catalog()
        print("\nHƯỚNG DẪN SỬ DỤNG NHANH:")
        print("1. Chạy thử nghiệm ngay với dữ liệu mẫu có sẵn trên GitHub:")
        print("   python src/download_datasets.py --prepare-sample")
        print("\n2. Xem danh mục các bộ dữ liệu đầy đủ: python src/download_datasets.py --list")
        print("\n3. Tải qua Roboflow:")
        print("   python src/download_datasets.py --source roboflow --dataset-id roboflow_smoking_behavior --api-key <YOUR_KEY>")
        print("\n4. Tải qua Kaggle:")
        print("   python src/download_datasets.py --source kaggle --dataset-id kaggle_smoking_detection\n")
        return

    if args.source == "roboflow":
        if not args.api_key:
            print("[LỖI] Vui lòng truyền --api-key <ROBOFLOW_API_KEY> hoặc gán biến môi trường ROBOFLOW_API_KEY.")
            sys.exit(1)
        download_from_roboflow(args.dataset_id, args.api_key, args.output_dir)

    elif args.source == "kaggle":
        download_from_kaggle(args.dataset_id, args.output_dir)


if __name__ == "__main__":
    main()
