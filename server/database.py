# -*- coding: utf-8 -*-
"""
数据库引擎与 Session 管理
========================
本平台只使用本地 SQLite 文件（<STORAGE_ROOT>/app.db），不再依赖 MySQL。
表结构由 ORM 模型通过 create_all 一次建全（模型即真相，无需增量迁移）。
"""

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from .config import settings


def _resolve_database_url() -> str:
    url = (settings.DATABASE_URL or "").strip()
    if url:
        return url
    db_file = Path(settings.STORAGE_ROOT).resolve() / "app.db"
    db_file.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_file.as_posix()}"


DATABASE_URL = _resolve_database_url()
IS_SQLITE = DATABASE_URL.startswith("sqlite")

if IS_SQLITE:
    # check_same_thread=False：FastAPI 线程池与后台训练线程会复用连接池中的连接
    # timeout=30：训练线程写 epoch 日志时，API 线程的写请求等待锁而非立刻报错
    engine = create_engine(
        DATABASE_URL,
        echo=settings.DEBUG,
        connect_args={"check_same_thread": False, "timeout": 30},
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")       # 读不阻塞写，训练时 API 照常响应
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()
else:
    # 兼容外接数据库的兜底（非本平台支持路径）
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_recycle=3600,
        echo=settings.DEBUG,
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """ORM 基类"""
    pass


def get_db():
    """FastAPI 依赖注入：获取数据库 Session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """建表（SQLite 下模型即完整表结构，不再有 MySQL 增量迁移）"""
    from . import models  # noqa
    Base.metadata.create_all(bind=engine)
    if not IS_SQLITE:
        print("[数据库] 警告：正在使用非 SQLite 数据库，本平台已去除 MySQL 增量迁移，"
              "表结构将以 ORM 模型为准直接 create_all。")
