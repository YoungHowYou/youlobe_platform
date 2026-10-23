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


def _repair_bigint_pk_tables() -> None:
    """修历史遗留的 SQLite 主键类型问题（自愈，启动时自动跑）。

    SQLite 只有精确的 `INTEGER PRIMARY KEY` 才是 rowid 别名（插入时自动生成）。
    早期版本把 `train_epoch_logs.id` 建成了 `BIGINT PRIMARY KEY`，导致每轮 epoch
    的 INSERT 都因 id 为 NULL 报 `NOT NULL constraint failed`——训练照常跑，
    但 epoch 日志一条都写不进，前端训练监控曲线全空、`current_epoch` 一直为 0。

    做法：建表语句里带 BIGINT 就把表 DROP 掉重建（这种表必然一行数据都没有，
    因为写入从来没成功过）；同时清理上次 DDL 中断可能留下的 `*_bak` 残留。
    """
    if not IS_SQLITE:
        return

    table = "train_epoch_logs"
    with engine.begin() as conn:
        # 清理上次中断留下的备份表（连带它携带的索引一起消失）
        conn.exec_driver_sql(f"DROP TABLE IF EXISTS {table}_bak")

        row = conn.exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not row or not row[0] or "BIGINT" not in row[0].upper():
            return                       # 表不存在（交给 create_all）或类型已正确

        rows = conn.exec_driver_sql(f"SELECT COUNT(*) FROM {table}").scalar()
        if rows:
            print(f"[数据库] 警告：{table}.id 是 BIGINT 主键但表里有 {rows} 行数据，"
                  f"跳过自动重建（请联系维护人员手工处理）")
            return

        print(f"[数据库] {table}.id 是 BIGINT 主键（SQLite 不自增），重建为 INTEGER")
        conn.exec_driver_sql(f"DROP TABLE {table}")     # 索引随表一起删除
        Base.metadata.tables[table].create(bind=conn)
        print(f"[数据库] {table} 已重建完成")


def _ensure_columns() -> None:
    """给已有 SQLite 库补新列（幂等，启动时自动执行）。

    `create_all` 只负责建**新表**，不会给已存在的表加列；而 SQLite 又支持
    `ALTER TABLE ... ADD COLUMN`。所以新增字段时把「表 → (列名, DDL)」写进
    下面的清单，老库启动时自动补齐，避免查询直接因缺列 500。
    新加的列必须是可空或带默认值的（SQLite 限制）。
    """
    if not IS_SQLITE:
        return

    wanted: dict[str, list[tuple[str, str]]] = {
        "annotations": [
            ("shape", "VARCHAR(16) NOT NULL DEFAULT 'polygon'"),
            ("shape_params", "JSON"),
        ],
    }

    with engine.begin() as conn:
        for table, columns in wanted.items():
            row = conn.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if not row:
                continue                     # 新库交给 create_all
            existing = {
                r[1] for r in conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
            }
            for name, ddl in columns:
                if name in existing:
                    continue
                conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
                print(f"[数据库] {table} 补列 {name}")


def init_db():
    """建表（SQLite 下模型即完整表结构，不再有 MySQL 增量迁移）"""
    from . import models  # noqa
    # 先修历史遗留：否则 create_all 可能撞上中断残留的孤儿索引，直接起不来
    _repair_bigint_pk_tables()
    Base.metadata.create_all(bind=engine)
    _ensure_columns()                        # 老库补新列（如 annotations.shape）
    if not IS_SQLITE:
        print("[数据库] 警告：正在使用非 SQLite 数据库，本平台已去除 MySQL 增量迁移，"
              "表结构将以 ORM 模型为准直接 create_all。")
