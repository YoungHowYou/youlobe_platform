# -*- coding: utf-8 -*-
"""
应用配置
========
通过环境变量或 .env 文件加载配置。
"""

from pathlib import Path
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ---- 数据库 ----
    # 本平台已去除 MySQL，默认使用本地 SQLite 文件（纯硬盘存储）。
    # 留空 → 自动使用 <STORAGE_ROOT>/app.db；如需外接其它库可自行填写 SQLAlchemy URL。
    DATABASE_URL: str = ""

    # ---- 文件存储 ----
    STORAGE_ROOT: str = str(Path(__file__).parent.parent / "storage")
    UPLOAD_DIR: str = "uploads"      # 相对于 STORAGE_ROOT
    DATASET_DIR: str = "datasets"
    RUNS_DIR: str = "runs"
    EXPORT_DIR: str = "exports"

    # ---- 服务 ----
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000", "http://localhost:80"]

    # ---- 日志轮转（logs/*.log，copytruncate）----
    LOG_MAX_MB: int = 20          # 单个日志文件超过这个大小就转
    LOG_BACKUP_COUNT: int = 5     # 保留几份 .1 / .2 …
    LOG_ROTATE_INTERVAL_S: int = 60

    # ---- 训练中间产物 ----
    # 训练结束（completed/cancelled）后自动删 raw/preprocessed/cropped/dataset
    # 与 weights/epochN.pt（保留 best.pt/last.pt/results.csv/args.yaml/plots/exports）；
    # 想保留现场调试就设 TRAIN_AUTO_PURGE_INTERMEDIATES=False
    TRAIN_AUTO_PURGE_INTERMEDIATES: bool = True

    # ---- Offline license ----
    LABELYOU_LICENSE_REQUIRED: bool = False
    LABELYOU_LICENSE_FILE: str = "config/license.dat"
    LABELYOU_WEB_DIST: str = ""

    # ---- 安全 ----
    # 留空 = 使用编译进代码的开发默认密钥（仅本地）；
    # 生产部署务必设置随机长串，auth_service 从 os.environ 读取，这里负责注入。
    JWT_SECRET: str = ""

    # ---- 图像 ----
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024   # 50MB
    THUMB_SIZE: int = 256                      # 缩略图最大边长

    class Config:
        env_file = ("config/.env", ".env")
        env_file_encoding = "utf-8"

    @property
    def upload_path(self) -> Path:
        p = Path(self.STORAGE_ROOT).resolve() / self.UPLOAD_DIR
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def dataset_path(self) -> Path:
        p = Path(self.STORAGE_ROOT).resolve() / self.DATASET_DIR
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def runs_path(self) -> Path:
        p = Path(self.STORAGE_ROOT).resolve() / self.RUNS_DIR
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()

# auth_service 通过 os.environ["JWT_SECRET"] 读取；把 .env 里的配置注入进去，
# 让"改 .env 即生效"对 JWT 密钥也成立（pydantic-settings 不会自动写 os.environ）。
if settings.JWT_SECRET:
    os.environ["JWT_SECRET"] = settings.JWT_SECRET
