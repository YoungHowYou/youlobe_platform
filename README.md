# youlobe_platform — 工业缺陷检测全栈平台

硅晶圆 / 方锭 / 光伏缺陷检测：Web 标注 + 模型训练 + 在线推理。

- **前端**：Vue 3 + Element Plus + Fabric.js（多边形/框/旋转框标注、6×6 分类网格、训练监控、在线推断）
- **后端**：FastAPI + SQLAlchemy
- **存储**：**只用本地硬盘** —— SQLite 文件库 `storage/app.db` + `storage/` 目录，**不需要 MySQL / Redis / MariaDB / Celery**
- **训练**：Ultralytics YOLO，跑在 uvicorn 进程内的守护线程里；**有 NVIDIA 显卡自动用 GPU，没有自动回落 CPU**

---

## 技术栈

### 数据层：只有 SQLite

| 项 | 说明 |
|---|---|
| 数据库 | **SQLite 单文件** `storage/app.db`（Python 内置 `sqlite3` 3.40.1，无需任何数据库驱动） |
| 连接字符串 | `server/config.py` 的 `DATABASE_URL` 默认为空 → `server/database.py` 自动拼 `sqlite:///<STORAGE_ROOT>/app.db` |
| ORM | SQLAlchemy 2.0.54（声明式 `DeclarativeBase` + `Mapped[]` 类型注解） |
| 建表 | `init_db()` 直接 `create_all`，模型即表结构真相，**没有迁移框架** |
| 并发调优 | `PRAGMA journal_mode=WAL` + `busy_timeout=30000` + `synchronous=NORMAL`，训练线程写 epoch 日志不阻塞 API 读 |
| 其它存储 | 图片/数据集/训练产物全是文件，落在 `storage/uploads|datasets|runs|exports` |

> 虚拟环境里**没有** `pymysql / redis / celery / kombu / billiard / psycopg` 任何一个包，代码中也没有对应 import —— 数据库部分确实只有 SQLite。

### 后端

| 组件 | 版本 | 用途 |
|---|---|---|
| FastAPI | 0.142.2 | HTTP API，自动生成 `/docs` Swagger |
| Starlette / Uvicorn | 1.7.0 / 0.54.0 | ASGI 框架与服务器（单进程，未开 `--reload`） |
| Pydantic / pydantic-settings | 2.13.5 / 2.15.0 | 请求校验 + `.env` 配置加载 |
| python-multipart | 0.0.32 | 文件上传（已 patch `MultiPartParser.max_files=100000` 支持批量导入） |
| PyYAML | 6.0.3 | 生成 `dataset.yaml`、读写模型 `metadata.yaml` |
| cryptography | 50.0.2 | 离线 license 的 Ed25519 验签 |
| 认证 | 自研 | JWT HS256 **手写实现**（不依赖 PyJWT）+ PBKDF2-HMAC-SHA256 10 万次迭代存密码 |
| 并发 | `threading` | 守护线程 + `TrainingDispatcher` 单例调度；取消靠轮询 DB 状态，**没有消息队列** |

### AI / 视觉

| 组件 | 版本 | 用途 |
|---|---|---|
| Ultralytics | 8.4.170 | YOLO 训练 / 推理 / 导出统一入口 |
| PyTorch + torchvision | 2.14.1+cu126 / 0.29.1+cu126 | 训练与 `.pt` 推理；CPU 机器换 cpu wheel |
| OpenCV | 5.0.0 | 读写图、形态学（膨胀/腐蚀三通道）、轮廓提取 |
| NumPy / Pillow | 2.2.6 / 12.3.0 | 数组运算、图像 IO |
| pandas / matplotlib | 2.3.3 / 3.10.9 | 解析 `results.csv`、画 Loss / 指标 / 学习率曲线 |
| 导出格式 | — | ONNX / OpenVINO / TensorRT（`core/export.py`） |
| 推理后端 | — | PyTorch `.pt` 必装；**OpenVINO 可选（当前未装）**，装上才有 OV 导出与 CPU/核显/NPU 推理 |

### 前端

| 组件 | 版本 | 用途 |
|---|---|---|
| Vue | 3.4 | `<script setup>` + TypeScript |
| Vue Router / Pinia | 4.3 / 2.1 | 路由与状态管理 |
| Element Plus | 2.7 | UI 组件库（`unplugin-auto-import` + `unplugin-vue-components` 按需自动引入） |
| Fabric.js | 5.3 | 标注画布：多边形 / 矩形 / 旋转框 |
| ECharts + vue-echarts | 5.5 / 6.7 | 训练监控曲线 |
| Axios | 1.7 | 请求封装与 token 注入 |
| Vite / TypeScript / vue-tsc | 5.2 / 5.4 / 2.0 | 构建；dev 端口 5174，`/api` 与 `/static/storage` 代理到 8000 |
| Node.js | v24.16.0 | 仅构建/开发需要，生产由后端托管 `web/dist` |

---

## 快速开始

```powershell
# 1. 建环境（自动按 nvidia-smi 选择 GPU / CPU 版 torch）
python tools\setup_env.py

# 2. 配置（数据库留空即用本地 SQLite）
copy .env.example .env      # 生产环境记得改 JWT_SECRET

# 3. 构建前端（构建后由后端托管，只需开一个端口）
cd web; npm install; npm run build; cd ..

# 4. 启动
venv\Scripts\python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8000
# 或双击 start_backend.bat
```

打开 <http://localhost:8000>，默认账号 `admin / admin123`。

开发前端另开：`cd web && npm run dev`（vite，端口 5174，热更新）。

---

## 部署为 Windows 服务（开机自启）

生产/产线机器建议注册成服务：开机即跑、不需要任何人登录、崩了自动重启、没有控制台窗口。

**以管理员身份**运行：

```powershell
deploy\install_service.bat        # 安装并启动（缺 pywin32 会自动装，并放行防火墙）
deploy\uninstall_service.bat      # 卸载（不动 storage\ 数据）
deploy\open_firewall.bat          # 单独放行/查看局域网访问（已含在 install 里）
```

| 项 | 值 |
|---|---|
| 服务名 / 显示名 | `LabelyouBackend` / Labelyou 缺陷标注平台后端 |
| 服务镜像 | `"<项目>\venv\Scripts\python.exe" "<项目>\tools\win_service.py" service` |
| 启动方式 | 自动（开机即启，SYSTEM 身份） |
| 崩溃恢复 | 5s → 10s → 60s 三次重试 |
| 日志 | `logs\service.log`（8MB 轮转 ×5）+ `logs\service.stdout.log` |
| 监听 | `0.0.0.0:8000`，可用 `LABELYOU_HOST` / `LABELYOU_PORT` 覆盖 |
| 管理 | `services.msc`、`sc query LabelyouBackend`、`net stop\|start LabelyouBackend` |
| 局域网访问 | 放行入站 TCP 8000（**仅本地子网**，公网来源仍拦截），规则名 `LabelyouBackend-8000` |

局域网内其它电脑直接开 `http://<本机IP>:8000`（本机 IP 用 `ipconfig` 或跑 `deploy\open_firewall.bat` 查看，
它会打印可用地址）。后端监听 `0.0.0.0`，不需要额外配置——**只差防火墙这一关**，
所以 `install_service.bat` 里已经包含了放行步骤。

> IP 由 DHCP 分配的话，续约后地址可能变。要长期固定访问地址，请在路由器上为这台机器做
> DHCP 保留（或设置静态 IP）。

实现是 `tools/win_service.py`（pywin32，**不需要下载任何第三方 exe**）。

> **服务镜像为什么是 venv 的 `python.exe` 而不是 pywin32 的 `pythonservice.exe`**：
> 后者按注册表定位**基础解释器**且不处理 `.pth`，连它引导时要用的 `servicemanager`
> 都导入不了（`servicemanager.pyd` 在 venv 里），表现为 SCM 报 7009 连接超时 + 日志空白。
> 直接用 venv 的解释器就没有这个问题。服务起不来时先跑 `win_service.py diag`，
> 它会打印 binPath 并直接指出是不是旧版注册。
前台调试不注册服务：`venv\Scripts\python.exe tools\win_service.py console`。

> 不想引入 pywin32 的话，`tools\start_service.bat` 顶部有现成的计划任务命令
> （`schtasks /SC ONSTART /RU SYSTEM`）；`deploy\install_services.bat` 是 NSSM 备选路径。

**改完代码要重启服务**（`server/`、`core/` 任意改动，服务没有 `--reload`）：

```powershell
net stop LabelyouBackend && net start LabelyouBackend
```

> 重启会把未跑完的训练任务标记为 `failed`——训练线程随进程结束，这是设计如此。

---

## 4 种项目类型

| task_type | 含义 | 数据集格式 | 默认模型 |
|---|---|---|---|
| `seg` | 实例分割 | YOLO polygon txt | yolo26s-seg / yolo11s-seg |
| `det` | 目标检测 | YOLO bbox txt | yolo26s.pt |
| `cls` | 图像分类 | ImageFolder（整图，不切片） | yolo11s-cls.pt（imgsz 固定 224） |
| `obb` | 旋转目标检测 | YOLO 4 角点 txt | yolo11s-obb.pt |

---

## 训练

- **设备**：训练页「训练设备」默认 `自动（优先 GPU）`。显式选 CPU 才是强制 CPU。
- **显存经验值**（8GB 卡）：seg/det/obb `imgsz=640` 用 `batch 8~16`；cls `imgsz=224` 可到 `32~64`。
  报 CUDA OOM 先把 batch 减半，其次降模型尺寸（s→n）。
- **并发**：同一时间只允许一个训练任务，第二个返回 409。
- **取消**：查库软取消，约 20 秒内在 epoch 边界生效；中断的任务其 `last.pt` / `best.pt` 仍在
  `storage/runs/task_<N>/runs/train/weights/`，可以被继承训练使用。
- **重启后端**：会把未跑完的任务标记为 `failed`（训练线程随进程结束）。

> 改了 `server/` 或 `core/` 下的文件必须重启 uvicorn（默认没开 `--reload`）。

---

## 运维工具

```powershell
venv\Scripts\python.exe tools\setup_env.py         # 一键建环境（GPU/CPU 自适应）
venv\Scripts\python.exe tools\check_gpu_train.py   # GPU 训练链路自检（合成数据 1 epoch）
venv\Scripts\python.exe tools\audit_storage.py     # 存储体检：僵尸项目 / 丢失图片 / 磁盘占用
venv\Scripts\python.exe tools\cleanup_storage.py   # 存储瘦身：推理缓存 / epoch 检查点
```

`audit_storage.py` 默认只体检不改动，清理要显式加参数：

```powershell
venv\Scripts\python.exe tools\audit_storage.py --purge-orphan-projects   # 删"图片文件全丢"的项目
venv\Scripts\python.exe tools\audit_storage.py --purge-project 3         # 删指定项目
venv\Scripts\python.exe tools\audit_storage.py --purge-missing-images    # 只删丢失的图片记录
venv\Scripts\python.exe tools\audit_storage.py --vacuum                  # 整理 app.db
```

---

## 数据备份 / 迁移

**只需要拷 `storage/` 目录**（`app.db` + 图片 + runs），不需要导 SQL，也不需要目标机装数据库。
`.pt` 权重在 `weights\` 或项目根目录，离线机器需一并拷过去。

---

## 目录结构

```
core/          train.py(训练) inference.py(滑窗推理) preprocess.py(形态学) sliding_window.py
server/        main.py database.py config.py models/ schemas/ routers/ services/ tasks/
web/           Vue 3 前端（npm run build → web/dist 由后端托管）
storage/       app.db · uploads/ · datasets/ · runs/ · exports/
tools/         建环境、自检、存储体检、离线打包、各类 eval / bench 脚本
docs/          离线部署手册、级联部署记录等
.github/skills/yolo-platform-deploy/   部署 skill（换机器部署照这个走）
```

