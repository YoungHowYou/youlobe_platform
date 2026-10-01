# CLAUDE.md — yolo26s_platform

工业缺陷检测全栈平台（硅晶圆/方锭/光伏）。Vue 3 前端 + FastAPI 后端 + SQLite 本地文件库 + 进程内训练线程 + Ultralytics YOLO。**已去除 MySQL/Redis/Celery**，全部状态存本地硬盘；训练仅支持 CPU。

---

## 启动

后端（单进程即可，训练任务在其内部的守护线程执行）：
```bash
# 在 D:\yolo26s_platform 目录
D:/yolo26s_platform/venv/Scripts/python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

前端：
```bash
cd web && npm run dev   # 默认 http://localhost:5174
```

环境变量（避免 ultralytics 联网装包卡住）：
```
YOLO_AUTOINSTALL=False
```

默认账号：`admin / admin123`。

---

## 改了代码必须重启的范围

| 改动 | 重启 |
|---|---|
| `server/routers/*` | uvicorn |
| `server/services/*`, `server/models/*`, `server/schemas/*` | uvicorn |
| `server/tasks/*`, `core/*` | uvicorn（训练线程跑在 uvicorn 进程内） |
| `web/src/*` | 不需要，vite hot reload 自动 |

uvicorn 没开 `--reload`，必须手动 kill PID + 重启。重启会把未跑完的训练任务标记为 failed（见"取消训练机制"）。

---

## 4 种任务类型

`projects.task_type` 决定整个流水线行为：

| type | 含义 | 训练数据集 | 模型默认 | 切片 |
|---|---|---|---|---|
| `seg` | 实例分割 | YOLO polygon txt | yolo26s-seg.pt | 大图 → resize → 滑窗 crop_size² |
| `det` | 目标检测 | YOLO bbox txt | yolo26s.pt | 同 seg |
| `cls` | 图像分类 | ImageFolder `dataset/<class>/` | yolo11s-cls.pt | **不切片**，整图 letterbox 到 imgsz |
| `obb` | 旋转目标检测 | YOLO 4 角点 txt（8 归一化坐标） | yolo11s-obb.pt | 同 seg |

cls 标签存 `images.class_id`（图级），其他 task 标签存 `annotations.polygon`（实例级）。

---

## cls 任务的几个坑（已修复，留作 reminder）

1. **imgsz 必须 224**：YOLO11-cls 在 ImageNet 224 预训练。`core/train.py` 已经强制 `if imgsz != 224: imgsz = 224`，并把 degrees/flipud/mosaic/copy_paste/mixup 全置 0。项目级 crop_size 对 cls 无意义。

2. **分类标签顺序与显示名彻底分离**：`prepare_classification_dataset` 使用 `0000/0001/...` 固定 ImageFolder label，训练后由 `classification_metadata.py` 把权重 `model.names` 写回项目显示名。继承训练会根据父模型类别语义（兼容 Crack/EdgeChip/Notch/OK）映射当前中文类别；不要再依赖 DB class_index 或目录字典序。OpenVINO 导出会把 `metadata.yaml` 写为 UTF-8 原始中文名。

3. **EasyLabel BMP 多是 128×128 单通道灰度**：cv2.imread 默认会扩成三通道，但训练/推理预处理要保持一致。

---

## 关键文件位置

```
core/
  train.py             — Ultralytics 训练入口，task_type 分支 + cancel_check 回调
  inference.py         — 滑窗推理（seg/det）
  preprocess.py        — Resize + 形态学（B=原图 G=膨胀 R=腐蚀 三通道）
  sliding_window.py    — 滑窗切割

server/
  main.py              — FastAPI 入口；monkey-patch starlette MultiPartParser.max_files=100000
  database.py          — Engine + 自动 ALTER TABLE 迁移列表（每次新加字段往这里追加）
  config.py            — settings.upload_path / runs_path / DATABASE_URL
  models/              — SQLAlchemy ORM
  schemas/             — pydantic
  routers/             — FastAPI 路由
  services/
    dataset_service.py — 4 种 task_type 的数据集准备（prepare_*_dataset）
    project_package.py — 项目导出/导入 zip
  tasks/train_task.py  — 训练流水线（纯函数）+ TrainingDispatcher 线程调度器，内含 cancel_check 注入

web/src/views/
  ProjectList.vue      — 项目管理（按 task_type 分组）
  ProjectDetail.vue    — 项目详情 + 编辑弹窗
  Annotator.vue        — 通用标注器（seg/det/obb 用）
  ClsAnnotator.vue     — cls 6×6 网格批量打标
  AnnotationConvert.vue — seg ⇆ det/obb 标注转换
  ImportProject.vue    — 多种导入入口（VOC/cls 文件夹）
  TrainConfig.vue      — 训练配置（参数缓存 + 继承训练自动加载）
  TrainMonitor.vue     — 训练监控（epoch 曲线 + 完整参数展示）
  InferenceView.vue    — 在线推断（队列 + 推理训练图 + 网格 + 切割小图）

tools/
  cleanup_storage.py   — 存储瘦身（推理缓存 + epoch checkpoints + 预处理产物）
  verify_cls.py        — cls 模型全量验证 + 混淆矩阵
  test_cls_pipeline.py — cls 端到端回归测试
```

---

## 推理 endpoint 路径速查

```
POST /api/inference/run                  上传图推理（multipart）
POST /api/inference/run-by-image-id      按 image_id 推理（不上传，跳本地读）
GET  /api/inference/project-images       项目图片抽样列表（推理训练图用）
POST /api/inference/crop-defects         seg 项目缺陷小图打 zip 下载
GET  /api/inference/models               模型下拉（含 cancelled 任务，文件存在性检查）
GET  /api/inference/devices              GPU 列表
GET  /api/inference/history              历史记录
PUT  /api/projects/{id}/train-config-cache    保存项目级训练参数缓存
GET  /api/projects/{id}/images/class-stats    cls 项目类别分布（项目全量）
PUT  /api/projects/{id}/images/batch-class    cls 批量打分类标签
```

---

## 训练任务可继承的状态

`completed` + `cancelled` 都允许（cancelled 任务硬盘上仍有 best.pt/last.pt，ultralytics 实时写盘）。
- 推断模型下拉、继承训练下拉两处都做了文件存在性检查
- cancelled 任务 label 加 `[取消@epoch N/M]` 标识
- best vs last 推荐：completed → best；cancelled 进度 < 30% → last（best 没收敛接着学）；≥ 30% → best

---

## 取消训练机制

`core/train.py` 注册三个 ultralytics 回调：
- `on_train_epoch_start`、`on_fit_epoch_end`：每 epoch 边界查 db
- `on_train_batch_end`：每 20 batch 查一次（throttle）

检测到 `task.status == 'cancelled'` 或任务被删 → `trainer.stop = True`，YOLO 在 epoch 边界优雅退出。最坏 ~20 秒生效。

`server/routers/train.py` delete 接口拒绝删 `pending/preparing/training/exporting`。服务重启时 `main.py` lifespan 会把上次未跑完的任务统一标记 failed。

## 训练并发模型（2026-09 去除 Celery 后）

- 训练在 uvicorn 进程内的**守护线程**执行（`TrainingDispatcher` 单例，同一时间只允许一个任务；第二个任务返回 409 并回滚记录）。
- 创建任务先 `db.commit()` 再 `dispatcher.start()`，启动失败会删除任务记录，不会留僵尸 pending（旧 Celery 文件队列的"永久排队"问题随之消除）。
- 取消仍是查库软取消（`cancel_check`），约 20 秒内于 epoch 边界生效；**不要**直接 kill 训练线程，会截断 best.pt 写盘。

---

## 训练内存 / Workers / 跨版本部署坑（2026-07-21，已按 CPU-only + 无 Celery 修订）

1. **不要再强制 `cache="ram"`**：4050 张 640×640 三通道切片解码后约占 4.6 GiB，CUDA OOM 自动重建 DataLoader 时耗尽系统内存，最后只显示 OpenCV `Failed to allocate 1228800 bytes`。`core/train.py` 现固定 `cache=False`；只影响读盘速度，不影响精度。
2. **看完整 traceback 的第一处 OOM**：最终的 OpenCV `Insufficient memory` 可能只是二次异常，真正原因常是更前面的内存耗尽。资源类失败由 `server/tasks/train_task.py` 转成中文参数建议，同时保留技术堆栈。
3. **Workers 给数据加载提速**：2026-07-04 提交 `0f25d3f` 把固定 `workers=0` 改成可配置。CPU 训练建议 2～8，Windows 异常时回到 0。
4. **训练线程跑在 uvicorn 进程内**：训练成功、失败、取消后都必须 `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()`（无 CUDA 时自动跳过）；已在任务 `finally` 统一执行。改了 `core/*` 或 `server/tasks/*` 后必须重启 uvicorn。
5. **`unexpected keyword argument 'workers'` 是混版，不是模型缺失**：说明新版 `server/tasks/train_task.py` 配了旧版 `core/train.py`。完整同步 `server/` + `core/`，确认 `inspect.signature(core.train.run_train)` 含 `workers`，再重启 uvicorn。复制 `.pt` 不能解决参数签名错误。
6. **训练强制 CPU**：`core/train.py` 入口会把任何非 cpu 的 device 参数强制改为 `cpu` 并打印告警（前端训练页也只提供 CPU 选项）；GPU 相关调优参数（TF32 等）保留但无实际效果。

---

## 离线部署包（2026-09 随去 Celery 重做）

- `tools/build_labelyou_offline_pack.py` 生成的离线包已改为：SQLite 本地库（`DATABASE_URL=` 留空，首次启动自动建 `storage\app.db`）、单 `start_api.bat`（训练线程在 API 进程内，不再启动 Worker）、不再内置 MariaDB 便携运行时。
- 部署/验收文档：`docs/labelyou_offline_deploy.md`。

---

## 项目38 两级级联部署记录（2026-07-29）

目标：在不牺牲一级高召回的前提下，用二级分类过滤倒角等一级假阳性。

1. 一级使用任务 68 的 `yolo11n-seg`，类别为 `隐裂 / 崩边 / 缺口`；二级使用任务 73 的 `yolo11n-cls`，类别索引必须固定为 `0=隐裂, 1=崩边, 2=缺口, 3=OK`。
2. 已验证的生产参数：一级 `resize=2048`、`tile=640`、`overlap=0.2`、`padding=32`、`conf=0.01`、NMS IoU=`0.5`、形态学三通道启用且膨胀/腐蚀核均为 `3`；二级只有在 `P(OK) >= 0.70` 时才否决一级候选，其他情况一律保留 NG。单图候选上限不能设为 20（测试中出现过 27 个候选），建议至少 50。
3. OpenVINO FP16 在 55 张人工复核生产图上的图级结果：OK 误报 `0/34`、NG 检出 `21/21`；381 个一级候选经二级后保留 204 个。该指标是图级验证，不等同于逐实例数量验收。
4. 同批数据上 FP16 与 FP32 的一级框、类别及二级 Top-1 完全一致；FP16 稳态总耗时 `0.788s/图`，FP32 为 `0.895s/图`，FP16 快约 11.9%。开发机是 i5-12450H，生产 i7-12700 必须另做实机 CT 验收。
5. 生产端若直接读取 `metadata.yaml` 文本，二级中文类别必须写成裸 UTF-8 字符：`0: 隐裂`、`1: 崩边`、`2: 缺口`、`3: OK`。不能使用 `\uXXXX` 转义，也不能加双引号；部分生产解析器会把它们当作类别名的一部分。只改 metadata 不改 `best.xml` / `best.bin`，无需重训。
6. 可复现工具：`tools/eval_cascade_openvino.py`（55 图级联测试）、`tools/export_cascade_review_crops.py`（导出二级实际小图）、`tools/run_single_cascade.py`（单图验证）。模型和部署 zip 均为存储产物，不提交 Git。

---

## 工作约定

1. **不要每次自动 `git push`**。只在用户明确说"提交/合并/push"时推。
2. **commit 风格**：`feat/fix/tune/tool: 中文描述`，message 带 `Co-Authored-By: Claude` 行。
3. **mask2png.py 是用户的 GUI 工具**，git status untracked 不要碰。
4. **`*.pt` 已加 .gitignore**，模型权重不入库。
5. **改 db 字段**：直接改 `server/models/*.py` 并重启 uvicorn，`create_all` 只负责建新表；SQLite 不支持 ALTER 加列的自动迁移，新字段对**新库**生效，老库需重建或手动 ALTER。
6. **cancel 任务后还想用其 best.pt**：直接去 `D:\yolo26s_platform\storage\runs\task_<N>\runs\train\weights\best.pt`，前端在线推断也能选到。

---

## 数据库

```
SQLite 本地文件：<STORAGE_ROOT>/app.db（默认 storage/app.db，首次启动自动建表）
```

- WAL 模式 + busy_timeout=30s：训练线程写 epoch 日志不阻塞 API 读。
- 排序 nullable 字段注意 SQLite 与 MySQL 差异，沿用 `func.coalesce(col_a, col_b)` 写法。

---

## 用户硬件 / 环境

- GPU: RTX 3060 6GB 移动版（同事相同）；训练仅 CPU
- OS: Windows 11 Pro for Workstations
- shell: git-bash（Bash 工具默认）—— Linux 风格命令
- 本机已部署运行实例：`D:\yolo26s_platform`（venv + SQLite `storage\app.db`），后端托管 `web\dist`，浏览器直接开 http://localhost:8000
- Node.js 便携版在 `D:\soft\nodejs\node-v22.14.0-win-x64`（未进系统 PATH，构建前端时临时加 PATH 或用绝对路径调 npm.cmd；vite dev 端口 5174）
- Python 3.14.7（per-user：`C:\Users\haoyu.yang\AppData\Local\Programs\Python\Python314`），venv 已迁移到 3.14 + torch 2.9.0+cpu（2026-09-21）。**系统级前置**：必须装 VC++ 2015-2022 Redist ≥14.40（`System32\MSVCP140.dll`），否则 torch ≥2.9 的 c10.dll 初始化失败（WinError 1114）——当年"锁死 2.8.0+3.10"的真凶就是它；离线部署目标机也要装 `vc_redist.x64.exe`。机器上仍保留 Python 3.10.11，旧环境在 `venv310-backup`（确认 3.14 稳定后可删）
- 开机自启：计划任务 `LabelyouBackend`（SYSTEM、开机触发+10s 延迟、失败 1 分钟自动重启、隐藏无控制台窗口），跑 `tools\start_service.bat`（日志在 `logs\backend.*.log`）。`shell:startup` 快捷方式已删除；手动开发仍可用 `start_backend.bat`
