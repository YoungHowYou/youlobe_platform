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
    # 每个 *_DIR 都可以写「相对 STORAGE_ROOT 的子目录」或「绝对路径」，
    # 方便把不同产物放到不同磁盘（大盘存原图、SSD 跑训练）：
    #   STORAGE_ROOT=/data/hdd/youlobe/storage    原图 / 数据集 / 导出（机械盘）
    #   RUNS_DIR=/data/ssd/youlobe/runs           训练工作区（SSD，读写最频繁）
    #   ARCHIVE_DIR=/data/hdd/youlobe/archive     训练结束后的结果归档（机械盘）
    STORAGE_ROOT: str = str(Path(__file__).parent.parent / "storage")
    UPLOAD_DIR: str = "uploads"      # 相对于 STORAGE_ROOT（或绝对路径）
    DATASET_DIR: str = "datasets"
    RUNS_DIR: str = "runs"
    EXPORT_DIR: str = "exports"

    # ---- 训练结果归档 ----
    # 训练结束（completed / cancelled）后把任务目录整体移动到 ARCHIVE_DIR，
    # 让 SSD 只保留正在跑的任务；关闭则任务目录留在 RUNS_DIR。
    TRAIN_AUTO_ARCHIVE: bool = True
    ARCHIVE_DIR: str = "archive"

    # ---- 数据库备份 ----
    # 定期把 app.db 备份到 <ARCHIVE_DIR>/db_backups（原图/标注在大盘之外时兜底）
    DB_BACKUP_ENABLED: bool = True
    DB_BACKUP_INTERVAL_H: float = 6.0
    DB_BACKUP_KEEP: int = 14

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
        return self._resolve_dir(self.UPLOAD_DIR)

    @property
    def dataset_path(self) -> Path:
        return self._resolve_dir(self.DATASET_DIR)

    @property
    def runs_path(self) -> Path:
        return self._resolve_dir(self.RUNS_DIR)

    @property
    def export_path(self) -> Path:
        return self._resolve_dir(self.EXPORT_DIR)

    @property
    def archive_path(self) -> Path:
        return self._resolve_dir(self.ARCHIVE_DIR)

    def _resolve_dir(self, value: str) -> Path:
        """相对路径 → <STORAGE_ROOT>/value；绝对路径原样使用（并自动建目录）。"""
        p = Path(value).expanduser()
        if not p.is_absolute():
            p = Path(self.STORAGE_ROOT).expanduser().resolve() / p
        p.mkdir(parents=True, exist_ok=True)
        return p.resolve()


settings = Settings()

# auth_service 通过 os.environ["JWT_SECRET"] 读取；把 .env 里的配置注入进去，
# 让"改 .env 即生效"对 JWT 密钥也成立（pydantic-settings 不会自动写 os.environ）。
if settings.JWT_SECRET:
    os.environ["JWT_SECRET"] = settings.JWT_SECRET
