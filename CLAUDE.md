# CLAUDE.md — yolo26s_platform

工业缺陷检测全栈平台（硅晶圆/方锭/光伏）。Vue 3 前端 + FastAPI 后端 + SQLite 本地文件库 + 进程内训练线程 + Ultralytics YOLO。**已去除 MySQL/Redis/Celery**，全部状态存本地硬盘；训练优先用 GPU（有 CUDA 走 GPU，无显卡机器自动回落 CPU）。

> 部署 / 换机器 / 排障请先看 **`.github/skills/yolo-platform-deploy/SKILL.md`**：
> 五步部署流程、服务化、局域网放行、僵尸数据清理，以及踩过的坑（服务宿主、批处理编码等）都在里面。

---

## 启动

后端（单进程即可，训练任务在其内部的守护线程执行）：
```bash
# 在项目根目录
python tools/setup_env.py      # 首次：建 venv + 装 torch（GPU/CPU 自适应）+ 后端依赖
venv/Scripts/python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8000
# 或直接双击 start_backend.bat
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
| `seg` | 实例分割 | YOLO polygon txt | yolo26s-seg.pt | **原图 → 按 crop_size/overlap 滑窗（不缩放）** |
| `det` | 目标检测 | YOLO bbox txt | yolo26s.pt | **原图 → 按 crop_size/overlap 滑窗（不缩放）** |
| `cls` | 图像分类 | ImageFolder `dataset/<class>/` | yolo11s-cls.pt | **不切片**，整图 letterbox 到 imgsz |
| `obb` | 旋转目标检测 | YOLO 4 角点 txt（8 归一化坐标） | yolo11s-obb.pt | **原图 → 按 crop_size/overlap 滑窗（不缩放）** |

cls 标签存 `images.class_id`（图级），其他 task 标签存 `annotations.polygon`（实例级）。

> **seg/det/obb 从 2026-10-22 起不再统一缩放**：数据集准备直接读原图（需要形态学时在整图做三通道），
> 再按 `crop_size` / `overlap` 滑窗。`projects.resize_h/w` 仅保留字段兼容，**不再参与训练**，
> 且训练任务的 `config` 里也不再写入这两个字段（2026-10-23 清理，训练监控页/参数导出不再出现“假的 Resize”）；
> 小于 `crop_size` 的图用 0 填充到窗口尺寸。推理端小图同样走 pad（不 resize），「长边缩放」建议保持 0。
> 切片标签规则：seg 保留顶点落在窗内的多边形；det/obb 只保留**中心落在窗内**的目标（det 裁成窗口内 bbox，obb 把 4 点裁到窗口），切片后过小的目标丢弃。
>
> **切片口径决策（2026-10-22）**：保持 overlap 滑窗 + 类内 NMS，**不引入 halo**。
> 当前 640×640 图、缺陷最长边 38px、模型输入 640 → 不切片，halo 无收益；
> 只有上大图且可重训时再评估（优先级：overlap+WBF > halo > ignore-region 训练）。

---

## cls 任务的几个坑（已修复，留作 reminder）

1. **imgsz 必须 224**：YOLO11-cls 在 ImageNet 224 预训练。`core/train.py` 已经强制 `if imgsz != 224: imgsz = 224`，并把 degrees/flipud/mosaic/copy_paste/mixup 全置 0。项目级 crop_size 对 cls 无意义。

2. **分类标签顺序与显示名彻底分离**：`prepare_classification_dataset` 使用 `0000/0001/...` 固定 ImageFolder label，训练后由 `classification_metadata.py` 把权重 `model.names` 写回项目显示名。继承训练会根据父模型类别语义（兼容 Crack/EdgeChip/Notch/OK）映射当前中文类别；不要再依赖 DB class_index 或目录字典序。OpenVINO 导出会把 `metadata.yaml` 写为 UTF-8 原始中文名。

3. **EasyLabel BMP 多是 128×128 单通道灰度**：cv2.imread 默认会扩成三通道，但训练/推理预处理要保持一致。

---

## 标注器区域变换（Annotator.vue）

- 「选择」工具：**拖区域本体=平移**、**拖绿色圆点=旋转**（按住 Shift 吸附 15°）、拖白点=改顶点。
- 任意工具下 **`Alt+拖拽` 区域本体也能平移**（不必先切「选择」工具），且不会误落新顶点/新笔迹。
- `Shift+方向键` 平移 1px（再按住 Ctrl 为 10px）；`[` / `]` 旋转 ∓15°（加 Shift 为 1°）；选中后工具栏出现 `-15°/+15°/+90°`。
- **改视口必须走 `canvas.setViewportTransform()`**：fabric 的 `findTarget` 用视口坐标（对象 `lineCoords`）做命中判定，
  旧版 `zoomFit()` 和「空格+拖拽」平移是直接改 `canvas.viewportTransform[4]/[5]` 的，
  结果「适应窗口 / 平移画布」之后所有标注区域都点不中、拖不动（表现为“标注完的 region 不能平移”）。
  以后任何改变视口的代码都必须经 `setViewportTransform()`（它会顺带刷新每个对象的 `setCoords`）。

---

## 关键文件位置

```
core/
  train.py             — Ultralytics 训练入口，task_type 分支 + cancel_check 回调
  inference.py         — 滑窗推理（seg/det）
  preprocess.py        — 形态学（B=原图 G=膨胀 R=腐蚀 三通道）；训练链已改 resize=False
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
  setup_env.py         — 一键建环境（按 nvidia-smi 自动选 GPU/CPU torch wheel）
  check_gpu_train.py   — GPU 训练链路自检（合成数据跑 1 epoch，不碰真实数据）
  audit_storage.py     — 存储体检/清理（僵尸项目、丢失的图片记录、app.db VACUUM）
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

## 训练模式（2026-10-23 增加第三种）

| train_mode | 含义 | 权重来源 | 记录字段 |
|---|---|---|---|
| `scratch` | 从官方预训练起步（按模型尺寸下拉） | ultralytics 自动下载/本地 `weights/` | `model_name`=尺寸名 |
| `finetune` | 继承本项目历史任务的 best/last | `storage/runs/task_<N>/...` | `inherited_from_task_id` |
| `custom` | **用户放服务器本地的任意 `.pt`** | `config.model_path` | `base_model_path`、`head_reinit`、`is_end2end` |

自定义模型的约束（`server/services/model_source.py`）：

- 只允许 `.pt`，且必须落在 `<仓库根>`、`<仓库根>/weights`、`<仓库根>/storage/models` 三个目录内（防止随意读盘），路径用绝对路径传给 `config.model_path`。
- `GET /api/train/local-models` 列出可选用模型（`path/name/size_mb/mtime`，按 mtime 倒序）；`GET /api/train/inspect-model?path=` 返回 `task/nc/names/end2end`（加载一次 YOLO 读元信息）。
- **任务类型不匹配 = 硬失败 400**（`detect` 模型不能进 `seg` 项目等）；**类别数不一致 = 只警告**，ultralytics 会重建检测/分割头、只迁移骨干，`config.head_reinit=true` 留痕。
- 前端训练页「模型尺寸」在 custom 模式下改为「自定义权重」显示，不再让人误以为尺寸下拉有效。

## 标注形状：矩形 / 圆形保持参数化（2026-10-23）

- `annotations.shape`：`polygon`（默认）/ `rect` / `circle`；`annotations.shape_params` 存参数（rect: `{x1,y1,x2,y2}` 归一化；circle: `{cx,cy}` 归一化 + `r` **像素**半径）。
- **`polygon` 仍是几何唯一真相**：矩形的 4 角点、圆的 48 点采样都写进 `polygon`，训练/推理/统计/项目导出全部只认 polygon，不需要认识形状类型；`shape_params` 只用于标注器参数化编辑，保存时由 polygon 反推（不维护第二份数据，平移/旋转/剪贴板/撤销无需特殊处理）。
- 标注器编辑：矩形给 4 个角点手柄（对角固定、轴对齐），圆形给「圆心（平移，半径不变）+ 半径手柄」；**不再给顶点手柄**（不会退化成拖一堆顶点）。
- 圆形工具有两种画法：「圆心+半径」和「**3点共圆**」（三点外接圆，共线时报错提示）。
- 矩形/圆形不支持旋转（`rotateSelectedBy` 直接提示），需要斜置矩形请用多边形工具。
- 老库迁移：`server/database.py::_ensure_columns()` 启动时用 `PRAGMA table_info` 比对并 `ALTER TABLE ... ADD COLUMN`（幂等），新增字段往这个清单里追加即可；新列必须可空或带默认值。

## 存储产物与删除口径（2026-10-23 统一收口）

磁盘产物都在这几处，**删除项目 / 删除训练任务必须走 `server/services/artifact_cleanup.py`**：

| 产物 | 路径 | 归属 |
|---|---|---|
| 原图 + thumbs | `storage/uploads/<pid>/` | 项目 |
| raw / preprocessed / cropped / dataset / runs(train) | `storage/runs/task_<id>/` | 任务 |
| 导出模型（ONNX/OpenVINO 目录） | `<task>/runs/train/exports/`（路径登记在 `exported_models.export_path`） | 任务 |
| 推理可视化图 | `storage/runs/inference/<id>_*.png` | 任务 |
| 导出模型下载缓存 | `<tmp>/model_downloads/<name>_id<record>.zip` | 任务 |
| 服务日志 | `logs/backend.log`（全局，不随项目/任务删） | — |

**日志轮转**（`server/services/log_rotation.py`，默认 20 MB × 5 份，`config/.env` 可用 `LOG_MAX_MB` / `LOG_BACKUP_COUNT` / `LOG_ROTATE_INTERVAL_S` 覆盖）：

- shell 重定向到 `logs/backend.log` 的情况用 **copytruncate**（拷成 `.1` → 旧的顺移 → 截断原文件），append fd 不用重开，进程无需重启。
- Windows 服务的 `logs/service.log` 由 `RotatingFileHandler` 自己转，worker 会识别 handler 的 `baseFilename` 并跳过，不会双重轮转。
- 运行监控页显示当前日志大小 + 轮转上限/份数/已有备份数。

- 删除项目会额外做**孤儿兜底**：`runs/task_<n>` / `uploads/<pid>` 里 DB 已无对应行的目录一并清掉（之前删失败的残留不会永久占盘）。
- 手动清理：`.venv/bin/python tools/audit_storage.py --purge-orphan-files --yes`（孤儿目录 + 推理图 + 下载缓存）。

### 训练中间产物（可自动回收）

训练结束（`completed` / `cancelled`）后会自动删掉**可再生**的大体积中间产物：
`raw/`、`preprocessed/`、`cropped/`、`dataset/`、`dataset.yaml`、`runs/*/weights/epoch*.pt`、`labels.cache`；
**保留** `best.pt` / `last.pt`（继承训练、导出、在线推理）、`results.csv`（曲线回填）、
`args.yaml`、`plots`、`exports/`、batch 预览图。

- 开关：`server/config.py` 的 `TRAIN_AUTO_PURGE_INTERMEDIATES`（`.env` 写 `False` 可关闭）。
- `failed` 任务不自动清（方便排查切片/标签问题），可手动清：
  训练监控页每行「清理中间产物」，或 `POST /api/train/tasks/{id}/purge-intermediates`。
- 实现：`server/services/artifact_cleanup.py::purge_task_intermediates`。

## 训练曲线（train_epoch_logs）

- `core/train.py` 的 `on_fit_epoch_end` 从 `trainer.tloss` 取训练集 loss。**ultralytics 8.3+ 里 `tloss` / `loss_items` 是 `{loss名: Tensor}` 的 dict**（老版是 Tensor 列表），按 Tensor 处理会静默失败 → `train_*_loss` 恒为 NULL、前端 Loss 曲线只剩 Val。
- YOLO26 的 loss 项是 `box/seg/cls/l1/sem`（**没有 dfl**），库里只存 box/seg/cls/dfl，所以前端 Loss 图按实际有数据的项动态渲染（不再出现空的 DFL 图）。
- `results.csv` 的 `epoch` 从 **1** 开始，而库里 `train_epoch_logs.epoch` 是 `trainer.epoch`（0 基）：回填/对齐必须 **-1**。
- **曲线断线的两个真因**（2026-10-23 修复）：
  1. 训练结束后 ultralytics 用 best.pt 再做一次 final val，此时 `trainer.epoch` 已自增到 `epochs`、`tloss` 还是最后一轮的值 → 早年会多写一条越界 epoch 行（曲线尾部多一个点、`current_epoch` 比 epochs 大 1 显示 110%）。现在 `ep >= total_epochs` 的回调直接忽略，且 upsert 不用 None 覆盖已有值。
  2. 某些 epoch 的 val loss 在 ultralytics 里是 `nan`（发散/未计算），入库变 NULL → 曲线中间断。现在 nan/inf 统一归为 SQL NULL，前端 `connectNulls: true` 跨空值连直线。
- 历史数据补全：`tools/backfill_epoch_logs.py`（已修正 -1 偏移）；接口层另有 `server/services/epoch_csv.py::fill_missing_train_loss`，读 `/train/tasks/{id}/epochs` 时从 `results.csv` 只补 NULL，不覆盖已有值。
- 验证指标（precision/recall/mAP、cls 的 top1/top5）**只有验证集**——ultralytics 每轮只在 val 集上评估，不存在训练集指标曲线。

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

## 手动早停（2026-10-22）

- `POST /api/train/tasks/{id}/stop`：仅 `training` 且 `current_epoch ≥ 1` 可用（`pending/preparing` 与第 0 个 epoch 一律 400）。
- 置 `status='stopping'` → `cancel_check` 在 epoch 边界返回 True → 收尾把状态改回 `completed`，
  并写 `config.stop_reason='manual'`、`config.stopped_at_epoch=N`。
- 与「取消」的区别：取消的终态是 `cancelled`（当作放弃）；早停按完成处理，导出/继承直接可用。
- 最坏等一个 epoch 生效（不做 batch 级加速）；`epochs` 字段不改写，前端显示「早停@N ep」。

## 训练并发模型（2026-09 去除 Celery 后）

- 训练在 uvicorn 进程内的**守护线程**执行（`TrainingDispatcher` 单例，同一时间只允许一个任务在跑）。
- **支持排队（FIFO）**：已有任务在跑时提交新任务不再报 409，而是进入内存队列（DB 里保持 `status='pending'`，接口返回 `queue_position`）；当前任务收尾后自动启动队列里的下一个。
  - 出队前会重新查库：任务被删 / 已被取消 / 状态已变 → 直接跳过，不会带着陈旧状态开跑。
  - 取消排队中的任务 = 从队列摘掉（立即生效，`finished_at` 当场写）；排队任务也可直接删除（`drop_queued` 后再删）。
  - 队列只在内存里：进程重启后 `main.py` lifespan 把遗留的 `pending` 标为 failed（错误信息提示“排队任务未开始执行，请重新创建”）。
- 创建任务先 `db.commit()` 再 `dispatcher.start()`。
- 取消仍是查库软取消（`cancel_check`），约 20 秒内于 epoch 边界生效；**不要**直接 kill 训练线程，会截断 best.pt 写盘。
- 回归测试：`.venv/bin/python -m unittest tests.test_training_queue -v`（假流水线，验证 FIFO、去重、退队、自动续跑）。

---

## 训练内存 / Workers / 设备选择（2026-07-21，已按 GPU/CPU 自适应 + 无 Celery 修订）

1. **不要再强制 `cache="ram"`**：4050 张 640×640 三通道切片解码后约占 4.6 GiB，CUDA OOM 自动重建 DataLoader 时耗尽系统内存，最后只显示 OpenCV `Failed to allocate 1228800 bytes`。`core/train.py` 现固定 `cache=False`；只影响读盘速度，不影响精度。
2. **看完整 traceback 的第一处 OOM**：最终的 OpenCV `Insufficient memory` 可能只是二次异常，真正原因常是更前面的内存耗尽。资源类失败由 `server/tasks/train_task.py` 转成中文参数建议，同时保留技术堆栈。
3. **Workers 给数据加载提速**：2026-07-04 提交 `0f25d3f` 把固定 `workers=0` 改成可配置。GPU 训练建议 4～8，纯 CPU 训练建议 2～8，Windows 异常时回到 0。
4. **训练线程跑在 uvicorn 进程内**：训练成功、失败、取消后都必须 `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()`（无 CUDA 时自动跳过）；已在任务 `finally` 统一执行。改了 `core/*` 或 `server/tasks/*` 后必须重启 uvicorn。
5. **`unexpected keyword argument 'workers'` 是混版，不是模型缺失**：说明新版 `server/tasks/train_task.py` 配了旧版 `core/train.py`。完整同步 `server/` + `core/`，确认 `inspect.signature(core.train.run_train)` 含 `workers`，再重启 uvicorn。复制 `.pt` 不能解决参数签名错误。
6. **设备选择走 `core/train.py::resolve_device()`**：`auto`（默认）→ 有 CUDA 用 `"0"`，没有就 `cpu`；显式传 `cpu` 才是强制 CPU。**不要再写死 `device="cpu"`**。

---

## GPU 训练（2026-10-01 实机部署，RTX 2080 8GB / 驱动 560.94 / CUDA 12.6）

- 环境：Python 3.10.11 + venv + `torch`/`torchvision` **cu126** wheel（`python tools/setup_env.py` 自动挑 index）。
  老驱动（<530）机器改用 `python tools/setup_env.py --torch-index cu121`。
- 前端训练页「训练设备」下拉数据来自 `/api/inference/devices`（后端返回 `total_mem_mb`），
  没有 CUDA 时只有「自动 / CPU」两项；缓存里失效的 GPU 编号（如换机器后的 `"1"`）自动回落 `auto`。
- 8GB 显存经验值：seg/det/obb `imgsz=640` 用 `batch 8~16`；cls `imgsz=224` 可到 `32~64`。
  报 CUDA OOM 时先把 batch 减半，其次是降模型尺寸（s→n）。
- `cudnn.benchmark=True` 只在 GPU 分支打开（imgsz 固定，收益稳定）；TF32 开关在 Turing 上无效但无害。

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
6. **cancel 任务后还想用其 best.pt**：直接去 `<项目根目录>\storage\runs\task_<N>\runs\train\weights\best.pt`，前端在线推断也能选到。

---

## 数据库

```
SQLite 本地文件：<STORAGE_ROOT>/app.db（默认 storage/app.db，首次启动自动建表）
```

- WAL 模式 + busy_timeout=30s：训练线程写 epoch 日志不阻塞 API 读。
- 排序 nullable 字段注意 SQLite 与 MySQL 差异，沿用 `func.coalesce(col_a, col_b)` 写法。

---

## 用户硬件 / 环境

- GPU: RTX 2080 8GB（驱动 560.94 / CUDA 12.6），训练默认走 GPU
- OS: Windows 11 Pro for Workstations
- shell: PowerShell（VS Code 集成终端）
- 本机已部署运行实例：本仓库根目录（`venv` + SQLite `storage\app.db`），后端托管 `web\dist`，浏览器直接开 http://localhost:8000
- Node.js v24.16.0（在系统 PATH 中，`node -v` / `npm -v` 直接可用；vite dev 端口 5174）
- Python 3.10.11（`C:\Users\yanghowyou\AppData\Local\Programs\Python\Python310`），venv 为 3.10 + `torch` cu126（2026-10-01 部署）。**系统级前置**：必须装 VC++ 2015-2022 Redist ≥14.40（`System32\MSVCP140.dll`），否则 torch 的 c10.dll 初始化失败（WinError 1114）；离线部署目标机也要装 `vc_redist.x64.exe`
- 开机自启：已注册 Windows 服务 `LabelyouBackend`（`deploy\install_service.bat` 安装，pywin32 实现，
  `tools\win_service.py`，SYSTEM 身份 + 自动启动 + 崩溃 5s/10s/60s 三次重试；日志 `logs\service.log`）；
  改完 `server/`/`core/` 后要 `net stop LabelyouBackend && net start LabelyouBackend`。
  前台调试用 `venv\Scripts\python.exe tools\win_service.py console`；
  手动开发也可用 `start_backend.bat`（仅后端）或 `start_all.bat`（后端 + vite dev）
