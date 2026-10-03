# -*- coding: utf-8 -*-
"""
一键环境安装（跨平台）
=====================
本平台只依赖本地硬盘（SQLite 文件库 + 文件存储），不需要 MySQL / Redis / MariaDB / Celery。

torch / torchvision 的 CPU 版与 GPU(CUDA) 版来自不同的 wheel index，PyPI 上的默认包是
数 GB 的 CUDA 大包。本脚本按机器是否有 NVIDIA 显卡自动选择：

    有 nvidia-smi → https://download.pytorch.org/whl/cu126  （GPU 训练/推理）
    无 nvidia-smi → https://download.pytorch.org/whl/cpu    （纯 CPU）

用法（项目根目录下执行）：
    python tools/setup_env.py                # 自动判断
    python tools/setup_env.py --torch-index cpu      # 强制 CPU 版
    python tools/setup_env.py --torch-index cu124    # 强制 CUDA 12.4 版（老驱动）

装完后启动：
    后端  start_backend.bat   或  venv\\Scripts\\python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
    前端  cd web && npm install && npm run build   （构建后由后端托管 http://localhost:8000）
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / "venv"
PY = VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

TORCH_INDEX = {
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu124": "https://download.pytorch.org/whl/cu124",
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cpu": "https://download.pytorch.org/whl/cpu",
}


def run(cmd: list) -> None:
    print(">>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.check_call([str(c) for c in cmd])


def has_nvidia() -> bool:
    return shutil.which("nvidia-smi") is not None


def main() -> int:
    parser = argparse.ArgumentParser(description="安装 yolo26s_platform 运行环境")
    parser.add_argument("--torch-index", choices=sorted(TORCH_INDEX), default=None,
                        help="torch wheel index；留空则按是否有 nvidia-smi 自动选择")
    args = parser.parse_args()

    # ---- 1) 虚拟环境 ----
    if PY.exists():
        print(f"[1/4] 复用虚拟环境：{VENV}")
    else:
        print(f"[1/4] 创建虚拟环境：{VENV}")
        run([sys.executable, "-m", "venv", str(VENV)])

    # ---- 2) PyTorch（必须在 requirements 之前装，否则 ultralytics 会从 PyPI 拉 CUDA 大包）----
    key = args.torch_index or ("cu126" if has_nvidia() else "cpu")
    print(f"[2/4] 安装 PyTorch（{key}）")
    run([PY, "-m", "pip", "install", "--upgrade", "pip"])
    run([PY, "-m", "pip", "install", "torch", "torchvision", "--index-url", TORCH_INDEX[key]])

    # ---- 3) 其余后端依赖 ----
    print("[3/4] 安装后端依赖 server/requirements.txt")
    run([PY, "-m", "pip", "install", "-r", str(ROOT / "server" / "requirements.txt")])

    # ---- 4) 校验 ----
    print("[4/4] 校验运行环境")
    run([PY, "-c", (
        "import torch, sys;"
        "print('python      :', sys.version.split()[0]);"
        "print('torch       :', torch.__version__);"
        "print('cuda build  :', torch.version.cuda);"
        "print('cuda usable :', torch.cuda.is_available());"
        "print('device      :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU');"
        "x = torch.randn(256, 256, device='cuda' if torch.cuda.is_available() else 'cpu');"
        "print('matmul ok   :', bool((x @ x).sum().isfinite()))"
    )])

    print("\n环境就绪。若 cuda usable / matmul ok 为 False，说明 GPU 驱动或 torch 版本不匹配，")
    print("可改用 --torch-index cu124 或 cu121 重新安装。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
