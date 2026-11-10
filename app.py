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
import socket
import argparse
import uvicorn

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Kiểm tra xem cổng port đã có tiến trình nào chiếm dụng chưa."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False

def find_available_port(preferred_port: int = 7860, max_attempts: int = 50) -> int:
    """Tự động dò tìm cổng port trống kế tiếp nếu cổng chỉ định đang bận."""
    port = preferred_port
    while is_port_in_use(port) and port < preferred_port + max_attempts:
        port += 1
    return port

def main():
    parser = argparse.ArgumentParser(description="Chạy Web Demo Hệ Thống Giám Sát Hút Thuốc Nơi Công Cộng")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Địa chỉ Host bind server (mặc định: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Cổng Port lắng nghe (mặc định: 7860)")
    parser.add_argument("--reload", action="store_true", help="Tự động reload code khi có thay đổi")
    args = parser.parse_args()

    # Kiểm tra và tự động giải phóng / chọn port khả dụng
    selected_port = args.port
    if is_port_in_use(selected_port):
        free_port = find_available_port(selected_port)
        print(f"\n[CHÚ Ý] Cổng {selected_port} đã bị chiếm bởi tiến trình khác.")
        print(f"       >> Hệ thống tự động chuyển sang cổng khả dụng: {free_port}\n")
        selected_port = free_port

    print("=" * 75)
    print(" HỆ THỐNG GIÁM SÁT & CẢNH BÁO HÀNH VI HÚT THUỐC LÁ NƠI CÔNG CỘNG")
    print(f" Web Server đang khởi động tại: http://localhost:{selected_port}")
    print(" Sẵn sàng kết nối Cloudflare Tunnel:")
    print(f"   cloudflared tunnel --url http://localhost:{selected_port}")
    print("=" * 75)

    uvicorn.run("src.server:app", host=args.host, port=selected_port, reload=args.reload)

if __name__ == "__main__":
    main()

