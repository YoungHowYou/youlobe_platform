# -*- coding: utf-8 -*-
"""
存储体检 / 清理工具
===================
解决「SQLite 里有记录、磁盘上图片文件已丢失」的僵尸数据问题（换机器、拷库没拷 uploads、
手工删过文件都会造成）。默认**只体检不改动**。

用法（项目根目录）:
    venv\\Scripts\\python.exe tools\\audit_storage.py                    # 体检，列出每个项目的文件完好度
    venv\\Scripts\\python.exe tools\\audit_storage.py --purge-orphan-projects
    venv\\Scripts\\python.exe tools\\audit_storage.py --purge-project 3
    venv\\Scripts\\python.exe tools\\audit_storage.py --purge-missing-images
    venv\\Scripts\\python.exe tools\\audit_storage.py --vacuum           # 整理 app.db，回收空间

    加 --yes 跳过二次确认。

清理范围（--purge-project / --purge-orphan-projects）:
    DB  : projects / defect_classes / images / annotations / train_tasks /
          train_epoch_logs / inference_results / exported_models 中该项目相关行
    磁盘: storage\\uploads\\<project_id>\\ 与 storage\\runs\\task_<task_id>\\
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.config import settings                      # noqa: E402
from server.database import SessionLocal, engine        # noqa: E402
from server.models import (                             # noqa: E402
    Annotation, DefectClass, ExportedModel, Image,
    InferenceResult, Project, TrainEpochLog, TrainTask,
)


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _img_exists(img: Image) -> bool:
    if not img.file_path:
        return False
    fp = Path(img.file_path)
    return (fp if fp.is_absolute() else settings.upload_path / fp).exists()


def print_db_stats(db) -> None:
    """整库行数 + storage 各子目录占用。"""
    from server.database import DATABASE_URL

    print(f"数据库：{DATABASE_URL}")
    db_file = Path(engine.url.database) if engine.url.database else None
    if db_file and db_file.exists():
        print(f"app.db 大小：{_human(db_file.stat().st_size)}")
        for suffix in ("-wal", "-shm"):
            side = Path(str(db_file) + suffix)
            if side.exists():
                print(f"  {side.name}：{_human(side.stat().st_size)}")

    tables = [
        ("projects", "项目"), ("defect_classes", "缺陷类别"), ("images", "图片"),
        ("annotations", "标注"), ("train_tasks", "训练任务"),
        ("train_epoch_logs", "epoch 日志"), ("inference_results", "推理记录"),
        ("exported_models", "导出模型"), ("users", "用户"),
    ]
    counts = []
    for table, label in tables:
        try:
            n = db.execute(__import__("sqlalchemy").text(f"SELECT COUNT(*) FROM {table}")).scalar()
        except Exception:
            n = "-"
        counts.append(f"{label} {n}")
    print("行数：" + " / ".join(counts))

    print("磁盘：")
    for name, path in (("uploads", settings.upload_path), ("datasets", settings.dataset_path),
                       ("runs", settings.runs_path),
                       ("exports", Path(settings.STORAGE_ROOT) / settings.EXPORT_DIR)):
        print(f"  {name:<9} {_human(_dir_size(path)):<10} {path}")
    print()


def audit(db) -> list:
    """返回 [{project, images, present, missing, annotations, tasks, disk}]。"""
    rows = []
    for proj in db.query(Project).order_by(Project.id).all():
        images = db.query(Image).filter(Image.project_id == proj.id).all()
        present = sum(1 for i in images if _img_exists(i))
        ann = (
            db.query(Annotation)
            .join(Image, Annotation.image_id == Image.id)
            .filter(Image.project_id == proj.id)
            .count()
        )
        tasks = db.query(TrainTask).filter(TrainTask.project_id == proj.id).all()

        disk = _dir_size(settings.upload_path / str(proj.id))
        for t in tasks:
            disk += _dir_size(settings.runs_path / f"task_{t.id}")

        rows.append({
            "project": proj,
            "images": len(images),
            "present": present,
            "missing": len(images) - present,
            "annotations": ann,
            "tasks": len(tasks),
            "disk": disk,
        })
    return rows


def print_report(rows: list) -> None:
    if not rows:
        print("数据库里没有任何项目。")
        return
    print(f"{'ID':>4}  {'名称':<24} {'类型':<5} {'图片':>6} {'完好':>6} {'缺失':>6} "
          f"{'标注':>6} {'训练':>5}  {'磁盘占用':>10}  状态")
    print("-" * 110)
    for r in rows:
        proj = r["project"]
        if r["images"] == 0:
            state = "空项目"
        elif r["present"] == 0:
            state = "★ 僵尸（文件全丢）"
        elif r["missing"] > 0:
            state = f"部分缺失（{r['missing']} 张）"
        else:
            state = "正常"
        name = (proj.name or "")[:22]
        print(f"{proj.id:>4}  {name:<24} {proj.task_type:<5} {r['images']:>6} {r['present']:>6} "
              f"{r['missing']:>6} {r['annotations']:>6} {r['tasks']:>5}  "
              f"{_human(r['disk']):>10}  {state}")
    print("-" * 110)
    total_disk = sum(r["disk"] for r in rows)
    zombie = [r["project"].id for r in rows if r["images"] > 0 and r["present"] == 0]
    empty = [r["project"].id for r in rows if r["images"] == 0]
    print(f"合计磁盘占用 {_human(total_disk)}；僵尸项目 {zombie or '无'}；空项目 {empty or '无'}")


def purge_projects(db, project_ids: list) -> None:
    for pid in project_ids:
        proj = db.query(Project).filter(Project.id == pid).first()
        if not proj:
            print(f"  [跳过] 项目 {pid} 不存在")
            continue

        tasks = db.query(TrainTask).filter(TrainTask.project_id == pid).all()
        task_ids = [t.id for t in tasks]
        images = db.query(Image).filter(Image.project_id == pid).all()
        image_ids = [i.id for i in images]

        if image_ids:
            db.query(Annotation).filter(Annotation.image_id.in_(image_ids)).delete(
                synchronize_session=False)
        if task_ids:
            db.query(TrainEpochLog).filter(TrainEpochLog.task_id.in_(task_ids)).delete(
                synchronize_session=False)
            db.query(ExportedModel).filter(ExportedModel.task_id.in_(task_ids)).delete(
                synchronize_session=False)
        db.query(InferenceResult).filter(InferenceResult.project_id == pid).delete(
            synchronize_session=False)
        db.query(Image).filter(Image.project_id == pid).delete(synchronize_session=False)
        db.query(DefectClass).filter(DefectClass.project_id == pid).delete(synchronize_session=False)
        db.query(TrainTask).filter(TrainTask.project_id == pid).delete(synchronize_session=False)
        db.delete(proj)
        db.commit()

        freed = 0
        up = settings.upload_path / str(pid)
        if up.exists():
            freed += _dir_size(up)
            shutil.rmtree(up, ignore_errors=True)
        for tid in task_ids:
            td = settings.runs_path / f"task_{tid}"
            if td.exists():
                freed += _dir_size(td)
                shutil.rmtree(td, ignore_errors=True)

        print(f"  [已删除] 项目 {pid}「{proj.name}」 图片 {len(images)} 张 / 训练任务 {len(tasks)} 个，"
              f"释放 {_human(freed)}")


def purge_missing_images(db) -> None:
    removed = 0
    for img in db.query(Image).all():
        if not _img_exists(img):
            db.query(Annotation).filter(Annotation.image_id == img.id).delete(
                synchronize_session=False)
            db.delete(img)
            removed += 1
    if removed:
        db.commit()
    print(f"  [已删除] 文件缺失的图片记录 {removed} 条（含其标注）")


def main() -> int:
    parser = argparse.ArgumentParser(description="存储体检 / 清理")
    parser.add_argument("--purge-orphan-projects", action="store_true",
                        help="删除所有「图片文件全丢」的僵尸项目")
    parser.add_argument("--purge-project", type=int, nargs="+", metavar="ID",
                        help="删除指定项目（可多个）")
    parser.add_argument("--purge-empty-projects", action="store_true",
                        help="删除没有任何图片的空项目")
    parser.add_argument("--purge-missing-images", action="store_true",
                        help="只删除文件已丢失的图片记录（保留项目）")
    parser.add_argument("--vacuum", action="store_true", help="VACUUM 整理 app.db")
    parser.add_argument("--yes", action="store_true", help="跳过确认")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        print_db_stats(db)
        rows = audit(db)
        print_report(rows)

        targets: list = []
        if args.purge_project:
            targets += args.purge_project
        if args.purge_orphan_projects:
            targets += [r["project"].id for r in rows if r["images"] > 0 and r["present"] == 0]
        if args.purge_empty_projects:
            targets += [r["project"].id for r in rows if r["images"] == 0]
        targets = sorted(set(targets))

        doing = targets or args.purge_missing_images or args.vacuum
        if not doing:
            print("\n（体检模式，未做任何改动）")
            return 0

        if not args.yes and targets:
            print(f"\n将删除项目：{targets}")
            if input("确认删除？输入 yes 继续：").strip().lower() != "yes":
                print("已取消。")
                return 0

        print()
        if targets:
            purge_projects(db, targets)
        if args.purge_missing_images:
            purge_missing_images(db)

        if args.vacuum:
            before = Path(engine.url.database).stat().st_size if engine.url.database else 0
            with engine.connect() as conn:
                conn.exec_driver_sql("VACUUM")
            after = Path(engine.url.database).stat().st_size
            print(f"  [已整理] app.db {_human(before)} → {_human(after)}")

        print("\n清理后：")
        print_db_stats(db)
        print_report(audit(db))
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
