# -*- coding: utf-8 -*-
"""语义分割权重复现校验。

用途：证明 ``core/semseg`` 的纯 PyTorch 复刻与实验 checkpoint 完全对齐。

典型用法（不要抢正在训练的 GPU，默认 CPU）：:

    # 1) 只做权重加载检查 + 单图指标
    python tools/verify_semseg.py \
        --ckpt weights/resunet34_512_e_best/best_mIoU_iter_2000.pth \
        --img-dir ../MMSegmentation_young/test_data/images \
        --mask-dir ../MMSegmentation_young/test_data/masks \
        --limit 10

    # 2) 全量并和实验 JSON 对比
    python tools/verify_semseg.py \
        --ckpt weights/resunet34_512_e_best/best_mIoU_iter_2000.pth \
        --img-dir ... --mask-dir ... \
        --compare-json weights/resunet34_512_e_best/test_eval_thr06.json \
        --out /tmp/verify_semseg.json

输出：逐图指标 + 汇总（Dice/IoU/Recall/InstRec/SkelRec/Frag/Boundary）。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.semseg import (  # noqa: E402
    build_model,
    evaluate_pair,
    find_checkpoint,
    load_weights,
    read_image_gray,
    sliding_window_predict,
    summarize,
)
from core.semseg.checkpoint import read_checkpoint_meta  # noqa: E402

IMAGE_SUFFIXES = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _list_stems(img_dir: Path):
    return {p.stem for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}


def _find_image(img_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = img_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _find_mask(mask_dir: Path, stem: str) -> Path | None:
    for suffix in (".png", ".bmp", ".tif", ".tiff"):
        candidate = mask_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="ResUNet34-clDice 语义分割权重校验")
    parser.add_argument("--ckpt", required=True, help="checkpoint 文件或实验 work_dir 目录")
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--mask-dir", required=True, help="GT mask 目录（像素值 0/实例ID）")
    parser.add_argument("--split", default=None, help="txt：每行一个图名（不含扩展名）")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--device", default="cpu", help="默认 cpu，避免抢占训练 GPU")
    parser.add_argument("--amp", action="store_true", help="CUDA 上启用 fp16（会引入微小数值差异）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 张，0=全部")
    parser.add_argument("--out", default=None, help="逐图结果 JSON 输出路径")
    parser.add_argument("--compare-json", default=None, help="实验 test_eval*.json，用于打印差值")
    parser.add_argument("--no-strict-load", action="store_true",
                        help="权重键不匹配时只告警（默认严格：missing/unexpected 直接报错）")
    args = parser.parse_args()

    ckpt = find_checkpoint(args.ckpt) or args.ckpt
    ckpt_path = Path(ckpt)
    if not ckpt_path.exists():
        print(f"[错误] checkpoint 不存在：{ckpt}")
        return 2

    print(f"[校验] checkpoint = {ckpt_path}")
    meta = {}
    try:
        meta = read_checkpoint_meta(ckpt_path)
        print(f"[校验] meta: iter={meta.get('iter')} epoch={meta.get('epoch')} "
              f"experiment={meta.get('experiment_name')}")
    except Exception as exc:               # meta 读不出来不影响权重
        print(f"[校验] meta 读取失败（忽略）：{exc}")

    model = build_model(num_classes=1, pretrained=False)
    load_weights(model, ckpt_path, strict=not args.no_strict_load)
    model.eval()
    print(f"[校验] 参数量 = {sum(p.numel() for p in model.parameters()):,}")

    img_dir = Path(args.img_dir)
    mask_dir = Path(args.mask_dir)
    if not img_dir.is_dir() or not mask_dir.is_dir():
        print(f"[错误] 目录不存在：{img_dir} / {mask_dir}")
        return 2

    if args.split:
        stems = [line.strip() for line in Path(args.split).read_text().splitlines()
                 if line.strip()]
    else:
        stems = sorted(_list_stems(img_dir) & _list_stems(mask_dir))
    if args.limit > 0:
        stems = stems[:args.limit]
    if not stems:
        print("[错误] 没有匹配到「图 + mask」样本")
        return 2

    print(f"[校验] 样本 {len(stems)} 张，window={args.window} stride={args.stride} "
          f"thr={args.threshold} device={args.device}")

    results = []
    started = time.time()
    for idx, stem in enumerate(stems, 1):
        img_path = _find_image(img_dir, stem)
        mask_path = _find_mask(mask_dir, stem)
        if img_path is None or mask_path is None:
            print(f"[跳过] {stem}: 图或 mask 缺失")
            continue
        img = read_image_gray(img_path)
        if img is None:
            print(f"[跳过] {stem}: 图像读取失败")
            continue
        gt_inst = read_image_gray(mask_path)
        if gt_inst is None:
            print(f"[跳过] {stem}: mask 读取失败")
            continue

        gt_binary = (gt_inst > 0).astype(np.uint8)
        has_instance = any(int(v) > 1 for v in np.unique(gt_inst))

        prob = sliding_window_predict(
            model, img,
            window=args.window, stride=args.stride,
            device=args.device, amp=args.amp,
        )
        pred = (prob > args.threshold).astype(np.uint8)
        row = evaluate_pair(pred, gt_binary, gt_inst if has_instance else None)
        row["id"] = stem
        results.append(row)

        print(f"[{idx}/{len(stems)}] {stem}: dice={row['dice']:.4f} "
              f"iou={row['iou']:.4f} rec={row['recall']:.4f} "
              f"skel={row['skeleton_recall']:.4f} frag={row['fragmentation']:.2f}")

    elapsed = time.time() - started
    summary = summarize(results)
    print("\n==== SUMMARY ====")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")
    print(f"耗时: {elapsed:.1f}s，平均 {elapsed / max(len(results), 1):.2f}s/图")

    compare = None
    if args.compare_json and Path(args.compare_json).exists():
        exp = json.loads(Path(args.compare_json).read_text(encoding="utf-8"))
        exp_summary = exp.get("summary", exp)
        print("\n==== 与实验 JSON 差值（平台 − 实验）====")
        compare = {}
        for key in summary:
            if key in exp_summary and np.isfinite(summary[key]):
                delta = summary[key] - float(exp_summary[key])
                compare[key] = {"platform": summary[key],
                                "experiment": float(exp_summary[key]),
                                "delta": delta}
                print(f"{key}: {summary[key]:.4f} - {exp_summary[key]:.4f} "
                      f"= {delta:+.4f}")

    if args.out:
        payload = {
            "checkpoint": str(ckpt_path),
            "meta": {k: v for k, v in meta.items() if isinstance(v, (str, int, float))},
            "args": vars(args),
            "summary": summary,
            "compare": compare,
            "per_image": results,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"\n结果已写入 {args.out}")

    # 实验最优点：test Dice 0.729（thr=0.6）
    dice = summary.get("dice", float("nan"))
    if np.isfinite(dice):
        print(f"\n[结论] Dice={dice:.4f}，实验基准 0.7290，"
              f"差值 {dice - 0.729:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
