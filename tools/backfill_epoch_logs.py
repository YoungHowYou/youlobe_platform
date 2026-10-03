# -*- coding: utf-8 -*-
"""
从 results.csv 回填训练曲线（epoch 日志）
========================================
使用场景：`train_epoch_logs` 一条都没写进去（例如该表的 id 被建成了
`BIGINT PRIMARY KEY`——SQLite 不会自增 → 每轮 INSERT 都报
`NOT NULL constraint failed`），但磁盘上的 `results.csv` 是完整的。
训练本身没问题，只是前端监控没数据。本工具把 CSV 里每一轮的指标补写回数据库，
并按 Ultralytics 的 fitness 规则重算 best_map50 / best_fitness。
运行时先调用 `init_db()`——它会自动把错误的主键类型修好（见 server/database.py），
所以「修表 + 回填」一条命令搞定。

用法（项目根目录）：
    venv\\Scripts\\python.exe tools\\backfill_epoch_logs.py            # 处理所有有结果的任务
    venv\\Scripts\\python.exe tools\\backfill_epoch_logs.py --task 1   # 只处理某个任务
    venv\\Scripts\\python.exe tools\\backfill_epoch_logs.py --dry-run  # 只统计不写入
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.database import SessionLocal, init_db          # noqa: E402
from server.models.project import Project                  # noqa: E402
from server.models.train_task import TrainEpochLog, TrainTask  # noqa: E402


def _task_type(db, task: TrainTask) -> str:
    """task_type 在 projects 表上（TrainTask 只有 project_id）。"""
    cfg = task.config or {}
    if cfg.get("task_type"):
        return cfg["task_type"]
    proj = db.query(Project).filter(Project.id == task.project_id).first()
    return (proj.task_type if proj else "seg")

# results.csv 列名 → train_epoch_logs 字段
FLOAT_MAP = {
    "train/box_loss": "train_box_loss",
    "train/seg_loss": "train_seg_loss",
    "train/cls_loss": "train_cls_loss",
    "train/dfl_loss": "train_dfl_loss",
    # cls 任务的单一 loss
    "train/loss": "train_cls_loss",
    "val/box_loss": "val_box_loss",
    "val/seg_loss": "val_seg_loss",
    "val/cls_loss": "val_cls_loss",
    "val/dfl_loss": "val_dfl_loss",
    "val/loss": "val_cls_loss",
    "metrics/precision(B)": "precision_b",
    "metrics/recall(B)": "recall_b",
    "metrics/mAP50(B)": "map50_b",
    "metrics/mAP50-95(B)": "map50_95_b",
    "metrics/mAP50(M)": "map50_m",
    "metrics/mAP50-95(M)": "map50_95_m",
    "metrics/accuracy_top1": "top1_acc",
    "metrics/accuracy_top5": "top5_acc",
    "lr/pg0": "lr",
}


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def find_results_csv(task: TrainTask) -> Path | None:
    """任务目录下找 results.csv（通常是 <output_dir>\\runs\\train\\results.csv）。"""
    if not task.output_dir:
        return None
    base = Path(task.output_dir)
    if not base.is_dir():
        return None
    hits = sorted(base.rglob("results.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = []
        for raw in reader:
            row = {(k or "").strip(): v for k, v in raw.items()}
            ep = _to_float(row.get("epoch"))
            if ep is None:
                continue
            fields = {"epoch": int(ep)}
            for col, field in FLOAT_MAP.items():
                if col in row:
                    val = _to_float(row[col])
                    # 同名字段（如 cls 的 train/loss）先到先得，别被空值覆盖
                    if val is not None and fields.get(field) is None:
                        fields[field] = val
            rows.append(fields)
    return rows


def backfill_task(db, task: TrainTask, dry_run: bool = False) -> dict:
    csv_path = find_results_csv(task)
    if csv_path is None:
        return {"task_id": task.id, "skipped": "没有 results.csv"}

    rows = read_rows(csv_path)
    if not rows:
        return {"task_id": task.id, "skipped": "results.csv 里没有数据行"}

    is_cls = _task_type(db, task) == "cls"
    is_seg = _task_type(db, task) == "seg"

    existing = {
        log.epoch: log
        for log in db.query(TrainEpochLog).filter(TrainEpochLog.task_id == task.id).all()
    }

    added = updated = 0
    for fields in rows:
        ep = fields["epoch"]
        if ep in existing:
            if not dry_run:
                for k, v in fields.items():
                    setattr(existing[ep], k, v)
            updated += 1
        else:
            if not dry_run:
                db.add(TrainEpochLog(task_id=task.id, **fields))
            added += 1

    # 按 Ultralytics 的 fitness 重算 best：
    #   Metric.fitness     = mAP@0.5:0.95        （权重 [0,0,0,1]）
    #   DetMetrics.fitness = box.fitness()
    #   SegmentationMetrics.fitness = seg.fitness() + box.fitness()
    #                             = mAP50-95(M) + mAP50-95(B)
    #   ClassifyMetrics    = top1
    best_ep, best_fit, best_map50 = None, -1.0, None
    for fields in rows:
        if is_cls:
            fit = fields.get("top1_acc") or 0.0
            m50 = fit
        else:
            fit = fields.get("map50_95_b") or 0.0
            if is_seg:
                fit += fields.get("map50_95_m") or 0.0
            m50 = fields.get("map50_b")
        if fit > best_fit:
            best_ep, best_fit, best_map50 = fields["epoch"], fit, m50

    last_ep = max(r["epoch"] for r in rows)
    if not dry_run:
        if task.current_epoch < last_ep + 1:
            task.current_epoch = last_ep + 1
        if best_ep is not None and best_fit > (task.best_fitness or 0):
            task.best_fitness = best_fit
            if best_map50 is not None:
                task.best_map50 = best_map50
        db.commit()

    return {
        "task_id": task.id,
        "csv": str(csv_path),
        "rows": len(rows),
        "added": added,
        "updated": updated,
        "epochs_covered": f"{rows[0]['epoch']}~{last_ep}",
        "best_epoch": best_ep,
        "best_fitness": round(best_fit, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="从 results.csv 回填训练曲线")
    parser.add_argument("--task", type=int, help="只处理指定任务 id")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写入")
    args = parser.parse_args()

    # 顺带修复 train_epoch_logs 的主键类型（幂等，无误则不动）
    init_db()

    db = SessionLocal()
    try:
        q = db.query(TrainTask).order_by(TrainTask.id)
        if args.task:
            q = q.filter(TrainTask.id == args.task)
        tasks = [t for t in q.all()]
        if not tasks:
            print("没有匹配的训练任务。")
            return 0

        for task in tasks:
            try:
                info = backfill_task(db, task, dry_run=args.dry_run)
            except Exception as exc:                        # noqa: BLE001
                db.rollback()
                print(f"任务 {task.id} 处理失败：{exc}")
                continue
            if "skipped" in info:
                print(f"任务 {task.id}：跳过（{info['skipped']}）")
            else:
                print(f"任务 {task.id}：{info['epochs_covered']} 共 {info['rows']} 轮 → "
                      f"新增 {info['added']} 条、更新 {info['updated']} 条；"
                      f"best fitness {info['best_fitness']} @ epoch {info['best_epoch']}")
                print(f"        来源 {info['csv']}")
        if args.dry_run:
            print("\n（dry-run，未写入）")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
