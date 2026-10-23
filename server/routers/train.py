# -*- coding: utf-8 -*-
"""
训练管理 API
"""

import logging
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models.defect_class import DefectClass
from ..models.project import Project
from ..models.train_task import TrainTask, TrainEpochLog
from ..schemas.train_task import TrainTaskCreate, TrainTaskOut, EpochLogOut
from ..services.epoch_csv import fill_missing_train_loss

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["训练管理"])


@router.post("/projects/{project_id}/train", response_model=TrainTaskOut, status_code=201)
def create_train_task(
    project_id: int,
    body: TrainTaskCreate,
    db: Session = Depends(get_db),
):
    """创建训练任务（异步执行）"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")

    # 将项目参数合并到训练配置中
    config = body.config.model_dump()
    config["resize_h"] = project.resize_h
    config["resize_w"] = project.resize_w
    config["crop_size"] = project.crop_size
    config["overlap"] = project.overlap

    # 自定义预训练模型：创建时就校验并在 config 里固化真实架构/类别信息
    if config.get("train_mode") == "custom":
        from ..services.model_source import inspect_local_model

        try:
            info = inspect_local_model(config.get("model_path"))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"自定义预训练模型不可用：{exc}")

        expected = {"seg": "segment", "det": "detect", "obb": "obb", "cls": "classify"}.get(project.task_type)
        if expected and info["task"] != expected:
            raise HTTPException(
                status_code=400,
                detail=f"模型任务类型为 {info['task']}，与项目类型 {project.task_type} 不匹配",
            )

        project_nc = db.query(DefectClass).filter(DefectClass.project_id == project_id).count()
        config["base_model_path"] = info["path"]
        config["model_name"] = info["stem"]
        config["is_end2end"] = info["end2end"]
        config["base_model_nc"] = info["nc"]
        # 类别数不一致不算错误：ultralytics 会重建检测/分割头，只迁移骨干权重
        config["head_reinit"] = bool(project_nc and info["nc"] != project_nc)

    task = TrainTask(
        project_id=project_id,
        task_name=body.task_name,
        config=config,
        epochs=config.get("epochs", 200),
        status="pending",
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    # 提交后台线程异步训练（无 Celery/Redis）。
    # 已有任务在跑时不再报 409，而是进队（FIFO）：当前任务收尾后自动接着跑下一个。
    # 想跳过排队可以传 queue=false（前端未使用，给脚本用）。
    from ..tasks.train_task import training_dispatcher
    started = training_dispatcher.start(task.id)
    task.queue_position = None if started else training_dispatcher.queue_position(task.id)
    if not started:
        print(f"[训练队列] task_{task.id} 已入队，前面还有 {task.queue_position} 个任务")

    return task


@router.get("/train/local-models")
def list_train_local_models():
    """列出服务器上的本地 .pt（weights/、仓库根目录、storage/models）。"""
    from ..services.model_source import list_local_models

    return list_local_models()


@router.get("/train/inspect-model")
def inspect_train_local_model(path: str):
    """读取本地 .pt 的任务类型 / 类别数 / 是否 end2end，供训练配置页预览。"""
    from ..services.model_source import inspect_local_model

    try:
        return inspect_local_model(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"无法读取模型: {exc}")


@router.get("/projects/{project_id}/train/tasks")
def list_train_tasks(project_id: int, db: Session = Depends(get_db)):
    """获取项目的训练任务列表"""
    project = db.query(Project).filter(Project.id == project_id).first()
    is_cls_project = bool(project and project.task_type == "cls")
    tasks = (
        db.query(TrainTask)
        .filter(TrainTask.project_id == project_id)
        # 加 id 做 tiebreaker：created_at 精度只到秒，并行提交的多个 task
        # 时间会完全相同，仅按时间排序会按插入顺序（正序）返回，最新的反而沉底。
        .order_by(TrainTask.created_at.desc(), TrainTask.id.desc())
        .all()
    )
    result = []
    from ..tasks.train_task import training_dispatcher

    for t in tasks:
        d = TrainTaskOut.model_validate(t).model_dump()
        # 排队位置（None 表示没在排队：已在跑或已结束）
        d["queue_position"] = training_dispatcher.queue_position(t.id)
        # 分类任务的最佳指标是 epoch 日志里的 Top-1，不是 train_tasks.best_map50。
        # 兼容旧表结构：继续复用 best_map50 输出字段，但在 API 中动态填入真实值。
        if is_cls_project:
            best_top1 = (
                db.query(func.max(TrainEpochLog.top1_acc))
                .filter(TrainEpochLog.task_id == t.id)
                .scalar()
            )
            if best_top1 is not None:
                d["best_map50"] = float(best_top1)
        # 从 config 中提取类别数
        if t.config and "class_names" in t.config:
            d["num_classes"] = len(t.config["class_names"])
        result.append(d)
    return result


@router.get("/train/tasks/{task_id}", response_model=TrainTaskOut)
def get_train_task(task_id: int, db: Session = Depends(get_db)):
    """获取训练任务详情"""
    task = db.query(TrainTask).filter(TrainTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    from ..tasks.train_task import training_dispatcher
    task.queue_position = training_dispatcher.queue_position(task.id)
    return task


@router.get("/train/tasks/{task_id}/epochs", response_model=list[EpochLogOut])
def get_epoch_logs(task_id: int, db: Session = Depends(get_db)):
    """获取训练 Epoch 日志（用于前端绘制训练曲线）

    注：旧版本训练（或仍在跑的旧进程）写入的 `train_*_loss` 是 NULL，
    这里从任务目录的 `results.csv` 兑底补上，前端 Loss 曲线才能同时
    画出 Train / Val 两条线（只补空值，不覆盖库里已有的数据）。
    """
    logs = (
        db.query(TrainEpochLog)
        .filter(TrainEpochLog.task_id == task_id)
        .order_by(TrainEpochLog.epoch)
        .all()
    )
    rows = [EpochLogOut.model_validate(log) for log in logs]
    if rows:
        task = db.query(TrainTask).filter(TrainTask.id == task_id).first()
        try:
            fill_missing_train_loss(task.output_dir if task else None, rows)
        except Exception as exc:      # 兑底失败不影响正常返回曲线
            print(f"[epochs] 从 results.csv 补训练集 loss 失败：{exc}")
    return rows


@router.post("/train/tasks/{task_id}/cancel")
def cancel_train_task(task_id: int, db: Session = Depends(get_db)):
    """友好取消训练任务。

    直接杀线程会在权重写盘半路截断文件，所以只把任务标记为 cancelled。
    core/train.py 中的 cancel_check 回调会在 epoch/batch 边界检测到状态变化，
    设置 trainer.stop=True，让 Ultralytics 自行收尾并保留 best.pt / last.pt。
    """
    task = db.query(TrainTask).filter(TrainTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    if task.status in ("completed", "failed", "cancelled"):
        raise HTTPException(status_code=400, detail=f"任务已处于终态: {task.status}")

    # 还在排队（未开始训练）的任务：直接从队列摘掉即可，立即生效
    from ..tasks.train_task import training_dispatcher
    if task.status == "pending" and training_dispatcher.drop_queued(task_id):
        task.status = "cancelled"
        task.finished_at = datetime.now()
        db.commit()
        return {"ok": True, "message": "已从训练队列中移除"}

    task.status = "cancelled"
    # 不立即写 finished_at：训练线程会检测 cancelled 并在收尾后写入真实结束时间。
    # 这样前端计时不会在 trainer 仍在保存 best/last 时提前停止。
    task.finished_at = None
    db.commit()
    return {"ok": True, "message": "已请求友好取消，训练将在最近检查点停止"}


@router.post("/train/tasks/{task_id}/stop")
def stop_train_task(task_id: int, db: Session = Depends(get_db)):
    """手动早停：在 epoch 边界优雅停止，但产物有效，最终记为 completed。

    与「取消」的区别：取消的终态是 cancelled（当作放弃）；早停的终态是 completed，
    并在 config 里记录 stop_reason='manual' / stopped_at_epoch，导出与继承直接可用。

    限制（与前端保持一致）：
      - 仅 status='training' 可用（pending/preparing 不允许）；
      - 至少完成 1 个 epoch（current_epoch >= 1），否则还没有可用权重；
      - 在 epoch 边界生效，最坏等一个 epoch。
    """
    task = db.query(TrainTask).filter(TrainTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    if task.status in ("completed", "failed", "cancelled"):
        raise HTTPException(status_code=400, detail=f"任务已处于终态: {task.status}")
    if task.status == "stopping":
        return {"ok": True, "message": "已在停止中，等待当前 epoch 结束"}
    if task.status != "training":
        raise HTTPException(
            status_code=400,
            detail=f"仅训练中的任务支持早停（当前状态: {task.status}）",
        )
    if int(task.current_epoch or 0) < 1:
        raise HTTPException(
            status_code=400,
            detail="尚未完成第一个 epoch，没有可用权重，暂时不能早停",
        )

    task.status = "stopping"
    # finished_at 留给训练线程在收尾时写入，避免计时提前停住
    task.finished_at = None
    db.commit()
    return {
        "ok": True,
        "message": f"已请求早停，将在第 {task.current_epoch} 个 epoch 结束后停止并保留模型",
    }


@router.delete("/train/tasks/{task_id}", status_code=204)
def delete_train_task(task_id: int, db: Session = Depends(get_db)):
    """删除训练任务（DB + runs/task_<id>/ + 关联 ExportedModel / InferenceResult）。

    ExportedModel.task_id 和 InferenceResult.task_id 没有声明外键（历史遗留），
    所以 ORM cascade 抓不到——手动清，否则会留孤儿行 + 磁盘上的 ONNX/推断图。
    """
    from ..models.exported_model import ExportedModel
    from ..models.inference_result import InferenceResult

    task = db.query(TrainTask).filter(TrainTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    # 持续中的状态不允许直接删除：必须先取消（取消会让 Ultralytics 在 epoch 边界停训）
    # 例外：task.status == 'pending' 且还在队列里（未真正开训）—— 直接从队列摘掉后删
    from ..tasks.train_task import training_dispatcher as _disp
    if task.status == "pending" and _disp.drop_queued(task_id):
        pass
    elif task.status in ("pending", "preparing", "training", "exporting"):
        raise HTTPException(status_code=400, detail=f"任务处于活跃状态 ({task.status})，请先取消再删除")

    # 抓出磁盘路径再删 DB（output_dir 可能为空，回退到 runs/task_<id>）
    run_dir: Path | None = None
    if task.output_dir:
        run_dir = Path(task.output_dir)
    else:
        run_dir = Path(settings.STORAGE_ROOT).resolve() / settings.RUNS_DIR / f"task_{task_id}"

    export_paths: list[str] = [
        e.export_path for e in db.query(ExportedModel)
            .filter(ExportedModel.task_id == task_id, ExportedModel.export_path.isnot(None))
            .all()
    ]
    inference_paths: list[str] = []
    for r in db.query(InferenceResult).filter(InferenceResult.task_id == task_id).all():
        for p in (r.original_path, r.overlay_path, r.overlay_morph_path, r.mask_path):
            if p:
                inference_paths.append(p)

    db.query(ExportedModel).filter(ExportedModel.task_id == task_id).delete(synchronize_session=False)
    db.query(InferenceResult).filter(InferenceResult.task_id == task_id).delete(synchronize_session=False)
    db.delete(task)
    db.commit()

    storage_root = Path(settings.STORAGE_ROOT).resolve()
    runs_root = (storage_root / settings.RUNS_DIR).resolve()
    try:
        if run_dir and run_dir.exists() and run_dir.resolve().is_relative_to(runs_root):
            shutil.rmtree(str(run_dir), ignore_errors=True)
    except Exception:
        log.exception("delete_train_task: failed to rm run_dir %s", run_dir)
    for ep_str in export_paths:
        try:
            ep = Path(ep_str).resolve()
            if ep.exists() and ep.is_relative_to(storage_root):
                if ep.is_file():
                    ep.unlink(missing_ok=True)
                elif ep.is_dir():
                    shutil.rmtree(str(ep), ignore_errors=True)
        except Exception:
            log.exception("delete_train_task: failed to rm export %s", ep_str)
    for url in inference_paths:
        try:
            rel = url.replace("/static/storage/", "", 1).lstrip("/\\")
            if not rel:
                continue
            target = (storage_root / rel).resolve()
            if target.exists() and target.is_relative_to(storage_root):
                target.unlink(missing_ok=True)
        except Exception:
            log.exception("delete_train_task: failed to rm inference file %s", url)


@router.get("/projects/{project_id}/train/completed-tasks")
def list_completed_tasks(project_id: int, db: Session = Depends(get_db)):
    """列出已完成的训练任务（供继承训练选择）"""
    from ..models.defect_class import DefectClass
    tasks = (
        db.query(TrainTask)
        .filter(TrainTask.project_id == project_id, TrainTask.status == "completed")
        .order_by(TrainTask.finished_at.desc())
        .all()
    )
    # 当前项目类别数
    current_nc = db.query(DefectClass).filter(DefectClass.project_id == project_id).count()

    result = []
    for t in tasks:
        prev_nc = len(t.config.get("class_names", [])) if t.config and "class_names" in t.config else None
        nc_match = (prev_nc == current_nc) if prev_nc is not None else None
        result.append({
            "task_id": t.id,
            "task_name": t.task_name,
            "best_map50": t.best_map50,
            "has_best": bool(t.best_model_path),
            "has_last": bool(t.last_model_path),
            "prev_nc": prev_nc,
            "current_nc": current_nc,
            "nc_match": nc_match,
            "finished_at": t.finished_at.isoformat() if t.finished_at else None,
        })
    return result


# ---------------------------------------------------------------------------
# GPU 实时状态（训练监控页：温度 / 显存 / 利用率 / 频率 / 功耗）
# ---------------------------------------------------------------------------

_GPU_SMI_FIELDS = (
    "index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,"
    "clocks.sm,clocks.mem,power.draw,power.limit,fan.speed"
)
_GPU_CACHE_TTL_S = 1.5          # 多个页面同时轮询时避免反复 shell 调 nvidia-smi
_GPU_CACHE: dict = {"ts": 0.0, "payload": None}


def _find_nvidia_smi() -> str | None:
    """定位 nvidia-smi（Windows 下 PATH 未配置时走默认安装路径）。"""
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    for candidate in (
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        r"C:\Windows\System32\nvidia-smi.exe",
    ):
        if Path(candidate).exists():
            return candidate
    return None


def _smi_num(value: str) -> float | None:
    """解析 nvidia-smi 的数值字段，[N/A] / NOT SUPPORTED 返回 None。"""
    text = (value or "").strip()
    if not text or text.upper() in {"N/A", "[N/A]", "NOT SUPPORTED", "[NOT SUPPORTED]"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_smi_line(line: str) -> dict | None:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 11:
        return None
    index = _smi_num(parts[0])
    used = _smi_num(parts[4])
    total = _smi_num(parts[5])
    return {
        "index": int(index) if index is not None else None,
        "name": parts[1],
        "temperature_c": _smi_num(parts[2]),
        "utilization_pct": _smi_num(parts[3]),
        "memory_used_mb": used,
        "memory_total_mb": total,
        "memory_pct": round(used / total * 100, 1) if used is not None and total else None,
        "clock_sm_mhz": _smi_num(parts[6]),
        "clock_mem_mhz": _smi_num(parts[7]),
        "power_w": _smi_num(parts[8]),
        "power_limit_w": _smi_num(parts[9]),
        "fan_pct": _smi_num(parts[10]),
    }


@router.get("/gpu/stats")
def gpu_stats():
    """读取 GPU 实时状态（nvidia-smi，1.5s 缓存）。

    无 NVIDIA 显卡/驱动时返回 available=False，前端隐藏卡片；
    仅读硬件传感器，不触碰 CUDA 上下文，对正在跑的训练无影响。
    """
    now = time.time()
    if _GPU_CACHE["payload"] is not None and now - _GPU_CACHE["ts"] < _GPU_CACHE_TTL_S:
        return _GPU_CACHE["payload"]

    payload = {
        "available": False,
        "reason": "",
        "sampled_at": datetime.now().isoformat(timespec="seconds"),
        "gpus": [],
    }
    exe = _find_nvidia_smi()
    if not exe:
        payload["reason"] = "未找到 nvidia-smi（无 NVIDIA 驱动或无显卡）"
    else:
        try:
            proc = subprocess.run(
                [exe, f"--query-gpu={_GPU_SMI_FIELDS}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if proc.returncode != 0:
                payload["reason"] = (proc.stderr or proc.stdout).strip()[:300] or "nvidia-smi 执行失败"
            else:
                gpus = [
                    g for g in (_parse_smi_line(l) for l in proc.stdout.splitlines() if l.strip()) if g
                ]
                payload["gpus"] = gpus
                payload["available"] = bool(gpus)
                if not gpus:
                    payload["reason"] = "nvidia-smi 未返回 GPU 信息"
        except Exception as exc:  # noqa: BLE001 - 任何驱动异常都不应影响 API
            payload["reason"] = f"读取 GPU 状态失败: {exc}"

    _GPU_CACHE["ts"] = now
    _GPU_CACHE["payload"] = payload
    return payload
