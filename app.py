#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Entrypoint: app.py
Mục đích: Khởi chạy Web Server & API phục vụ Demo Thuyết trình & Cloudflare Tunnel.
Cách dùng:
    python app.py
    python app.py --port 8000 --host 0.0.0.0
================================================================================
"""

import sys
import argparse
import uvicorn

def main():
    parser = argparse.ArgumentParser(description="Chạy Web Demo Hệ Thống Giám Sát Hút Thuốc Nơi Công Cộng")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Địa chỉ Host bind server (mặc định: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Cổng Port lắng nghe (mặc định: 8000)")
    parser.add_argument("--reload", action="store_true", help="Tự động reload code khi có thay đổi")
    args = parser.parse_args()

    print("=" * 75)
    print(" HỆ THỐNG GIÁM SÁT & CẢNH BÁO HÀNH VI HÚT THUỐC LÁ NƠI CÔNG CỘNG")
    print(f" Web Server đang khởi động tại: http://localhost:{args.port}")
    print(" Sẵn sàng kết nối Cloudflare Tunnel:")
    print(f"   cloudflared tunnel --url http://localhost:{args.port}")
    print("=" * 75)

    uvicorn.run("src.server:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
