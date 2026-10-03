---
name: yolo-platform-deploy
description: 在 Windows（NVIDIA 显卡可选）机器上从零部署 / 迁移 / 排障 yolo26s_platform（工业缺陷检测全栈平台）。覆盖 SQLite-only 本地化（彻底去掉 MySQL/Redis/Celery）、GPU 训练环境（torch cu126 + 自动选卡）、前端构建与后端托管、注册成开机自启的 Windows 服务、局域网放行、僵尸数据清理，以及部署过程中踩过的所有坑。当用户说"部署这个平台""换机器部署""去掉 MySQL/Redis""让训练走显卡""做成服务/开机自启""局域网打不开""平台起不来/训练报错"时使用。
---

# yolo26s_platform 部署 skill

> **本机实测基线**（2026-10-01）：Windows 11 + RTX 2080 SUPER 8GB（驱动 560.94 / CUDA 12.6）
> + Python 3.10.11 + torch 2.14.1+cu126 + ultralytics 8.4.170 + Node 24。
> 部署在 `<项目根>`，Windows 服务名 `LabelyouBackend`，监听 8000。

---

## 0. 架构与设计前提

**一句话**：单个 uvicorn 进程同时承担 API + 前端静态托管 + 训练执行；状态全部落本地硬盘；
训练有 CUDA 走 GPU、没有自动回落 CPU。**没有 MySQL / Redis / MariaDB / Celery / Worker。**

```
浏览器 ──→ uvicorn :8000（0.0.0.0，同时托管 web/dist）
              ├─ SQLAlchemy → storage/app.db        (SQLite + WAL)
              ├─ storage/{uploads,datasets,runs,exports}
              └─ TrainingDispatcher → 守护线程 → core/train.py → Ultralytics → CUDA:0
```

三个"刻意为之"的设计，改动前先理解：

1. **训练跑在 API 进程内的守护线程里**（`server/tasks/train_task.py`），不是独立 worker。
   好处是零中间件；代价是同一时刻只允许一个训练任务（第二个返回 409），且**重启服务会中断训练**
   ——下次启动时 `main.py` lifespan 会把未完成任务统一标成 `failed`。
2. **SQLite 单文件、没有迁移框架**：`init_db()` 直接 `create_all`，**只建新表、不会给老表加列**。
   加了字段的老库要么重建，要么手动 `ALTER TABLE`。
3. **服务镜像就是 venv 的 `python.exe`**，不是任何第三方服务包装器（原因见 §6.1，别改回去）。

---

## 1. 部署五步（TL;DR）

| # | 做什么 | 命令 |
|---|---|---|
| 1 | 前置检查 | `nvidia-smi` / `py -0p` / `node -v` |
| 2 | 建环境 | `python tools\setup_env.py` |
| 3 | 构建前端 | `cd web` → `npm install` → `npm run build` |
| 4 | 装成开机自启服务 | **管理员**运行 `deploy\install_service.bat`（含防火墙放行） |
| 5 | 验收 | `venv\Scripts\python.exe tools\win_service.py wait-health` |

装完局域网其它机器即可访问 `http://<本机IP>:8000`（默认 `admin / admin123`）。
下面各节是每一步的展开，§6 是踩坑记录——**换机器部署前至少看一遍 §6.1**。

---

## 2. 建环境

### 2.1 前置检查

```powershell
nvidia-smi                              # 有输出 = 有 N 卡，记下驱动版本
py -0p                                  # 可用的 Python 解释器
node -v ; npm -v                        # 前端构建需要 Node 18+
Test-NetConnection pypi.org -Port 443   # 离线机器需另备 wheel 包
```

按驱动选 torch wheel：

| 条件 | 选择 |
|---|---|
| `nvidia-smi` 驱动 ≥ 530 | `cu126`（推荐） |
| 驱动 < 530 | `--torch-index cu121` |
| 无 `nvidia-smi` | `--torch-index cpu`，其余流程完全相同 |

**系统级前置**：VC++ 2015-2022 Redist ≥ 14.40（`C:\Windows\System32\MSVCP140.dll`）。
缺了会在 `import torch` 时报 `WinError 1114 / c10.dll 初始化失败`——报错完全看不出是这个原因。

### 2.2 一键建环境

```powershell
cd <项目根目录>
python tools\setup_env.py                  # 自动按 nvidia-smi 选 GPU/CPU wheel
python tools\setup_env.py --torch-index cu121   # 老驱动
python tools\setup_env.py --torch-index cpu     # 无显卡
```

**顺序很关键**：先装 torch（走专用 wheel index）再装 `server/requirements.txt`。
反过来会让 ultralytics 从 PyPI 拉一个数 GB 的默认 CUDA 包，白下一遍
（所以 `requirements.txt` 里**故意不写** torch/torchvision）。

### 2.3 验收

```powershell
venv\Scripts\python.exe -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望：2.14.1+cu126  12.6  True  NVIDIA GeForce RTX 2080 SUPER
```

真跑一遍训练链路（合成数据 1 个 epoch，**不碰真实项目数据**）：

```powershell
venv\Scripts\python.exe tools\check_gpu_train.py
# 期望结尾：[OK] 训练链路可用（1 epoch 完成）
# 期望中间：Ultralytics ... torch-2.14.1+cu126 CUDA:0 (NVIDIA GeForce RTX 2080 SUPER, 8192MiB)
```

> `torch.cuda.get_arch_list()` 里应有 `sm_75`（Turing 卡）。wheel 自带 CUDA runtime，
> **不需要额外安装 CUDA Toolkit**。

### 2.4 预训练权重（离线机器必看）

Ultralytics 首次训练会从 GitHub 下载 `.pt`，缓存位置是**项目根目录的 `weights\`**（或项目根目录本身）。
离线机器必须提前把这些权重拷过去，否则首次训练会一直卡在 `Downloading ...`。

---

## 3. 前端构建

后端（`server/main.py`）会自动托管 `web/dist`，构建一次之后目标机不需要 Node：

```powershell
cd web
npm install
npm run build        # 产出 web/dist，被 SPA 兜底路由接管
```

- `npm run build` = `vue-tsc && vite build`；历史类型错误会中断构建，此时可退化为 `npx vite build`。
- 构建过就只留 8000 一个端口，最省事。开发时才 `npm run dev`（vite，5174，`/api` 与 `/static/storage` 代理到 8000）。
- 页面 404 / 空白 = `web/dist` 没构建。

---

## 4. 注册成开机自启的 Windows 服务

产线机器不要用 `start_backend.bat`（要有人登录 + 保持窗口开着）。注册成服务 = 开机即跑、无需登录、
崩溃自动重启、无控制台窗口。

```powershell
# 以【管理员】身份运行
deploy\install_service.bat       # 装 + 起 + 防火墙放行 + /api/health 自检，缺 pywin32 会自动装
deploy\uninstall_service.bat     # 卸载（不动 storage\ 数据，同时移除防火墙规则）
```

| 项 | 值 |
|---|---|
| 服务名 / 显示名 | `LabelyouBackend` / Labelyou 缺陷标注平台后端 |
| 运行身份 | `LocalSystem`，启动方式 `AUTO_START` |
| 服务镜像（binPath） | `"<项目>\venv\Scripts\python.exe" "<项目>\tools\win_service.py" service` |
| 崩溃恢复 | `sc failure`：5s → 10s → 60s 三次重试 |
| 日志 | `logs\service.log`（8MB 轮转 ×5）+ `logs\service.stdout.log` |

**改完代码必须重启服务**（服务不带 `--reload`）：

```powershell
net stop LabelyouBackend; net start LabelyouBackend
```

### 4.1 排查用的五个子命令

```powershell
venv\Scripts\python.exe tools\win_service.py console       # 前台跑同一套代码，Ctrl+C 退出（调试首选）
venv\Scripts\python.exe tools\win_service.py diag          # 一键诊断：状态/binPath/事件日志/日志尾部
venv\Scripts\python.exe tools\win_service.py wait-health   # 轮询 /api/health 最多 60s，就绪 0 / 超时 1
venv\Scripts\python.exe tools\win_service.py free-port     # 杀掉占用 8000 的进程
venv\Scripts\python.exe tools\win_service.py open-firewall # 放行 8000（管理员）
```

安装脚本启动失败时会自动调用 `diag`。

### 4.2 服务化本身的五个关键点

1. **必须 `os.chdir(项目根)`**——服务工作目录默认是 `C:\Windows\System32`，而 `.env` 里
   `STORAGE_ROOT=./storage` 是相对路径，不切目录会写到 System32 下；同时把项目根塞进 `sys.path`，
   否则 `from server.main import app` 直接 ImportError。
2. **uvicorn 跑在子线程**，主线程 `WaitForSingleObject(stop_event)` 等 SCM 停止信号。
   子线程中 uvicorn 会自动跳过 signal handler 安装，服务环境正好没有控制台信号。
3. **日志必须落文件**：服务没有 stdout。用 `log_config=None` 让 uvicorn 的 logger 冒泡到 root，
   root 挂 `RotatingFileHandler`；同时把 `sys.stdout/stderr` 重定向到 `service.stdout.log`——
   否则 `sys.stdout` 为 `None`，**任何库的一次 `print` 都会抛异常**。
4. **`SvcStop` 先上报 `SERVICE_STOP_PENDING`** 再关服务，否则 SCM 等 30 秒超时。
5. **装之前先腾出 8000 端口**（`free-port`），手动起的实例还占着端口的话服务起不来。

---

## 5. 局域网访问

后端监听 `0.0.0.0`，服务装好本来就能被局域网访问，**唯一障碍是 Windows 防火墙**：
三个配置文件默认开启且入站默认拦截。

```powershell
# 以管理员身份运行（install_service.bat 已内含这一步）
deploy\open_firewall.bat
```

- 规则名 `LabelyouBackend-8000`，**只放行 `LocalSubnet`**（公网来源仍拦截，这是有意的）；
  卸载脚本会一并移除。
- 跑完会打印本机可用的访问地址，其它电脑直接开 `http://<本机IP>:8000`。
- **IP 由 DHCP 分配时续约后会变** → 建议在路由器上给这台机器做 DHCP 保留，否则同事存的链接会失效。
- 本机有多个网卡时，`169.254.x.x` 是没插网线的自动地址，**别用它**。

---

## 6. 踩过的坑（换机器前必读）

### 6.1 ★ 服务镜像不要用 pywin32 的 `pythonservice.exe`（改了 3 轮才定位）

- **现象**：`net start` 失败；`sc query` 显示 `STOPPED` 且**退出码 0**
  （新建但从未成功启动的服务也是这状态，别被退出码误导）；`logs\service.log` **一行都不写**；
  System 日志是 **7009「等待服务连接超时」**（即 1053）。
- **真因**：`pythonservice.exe` 按注册表 `PythonCore\<ver>\InstallPath` 定位解释器，用的是
  **基础解释器**且**不处理 `.pth`** → `win32` 目录不在 `sys.path` 上 →
  它引导时自己 `import servicemanager` 就 `ModuleNotFoundError`（该模块在
  `venv\Lib\site-packages\win32\`），进程在加载我们的模块**之前**就死了。
- **取证关键**：错误只写在 Application 日志的 `Python Service` 事件里，且消息文本取不到
  （显示"无法检索事件消息文本"），必须读**插入字符串**：
  ```powershell
  Get-WinEvent -LogName Application -MaxEvents 300 |
    Where-Object { $_.ProviderName -eq 'Python Service' } |
    ForEach-Object { $_.TimeCreated; $_.Properties | ForEach-Object { $_.Value } }
  ```
- **试过但无效的四条路（别再重复踩）**：
  ① 在自己模块里 patch `sys.path`——太晚，宿主 import 在前；
  ② 把 `python3XX.dll` 等运行库拷到宿主旁——与该问题无关；
  ③ 把 pywin32 的 .py/.pyd 拷进 `venv\`——那目录要等加载我们模块后才进 `sys.path`；
  ④ 注册表 `PythonCore\3.10\PythonPath`——现代 CPython 已不再读它。
  另外 `PYTHONPATH` 环境变量**确实有效**（实测能起），但服务拿不到：SCM 在开机时缓存环境块，
  改完要重启才生效，且会污染全机所有 Python 进程。
- **最终方案**：服务镜像直接用 venv 的解释器，绕开整个宿主问题：
  ```
  binPath = "<项目>\venv\Scripts\python.exe" "<项目>\tools\win_service.py" service
  ```
  `service` 子命令做三件事：`servicemanager.Initialize()` →
  `PrepareToHostSingle(BackendService)` → `StartServiceCtrlDispatcher()`（阻塞）。
  零环境变量、零第三方 exe、不用重启。
- **顺带记住**：`ServiceFramework.SvcRun()` 是**先上报 `SERVICE_RUNNING` 再调 `SvcDoRun()`**，
  所以 uvicorn 启动慢也不会 1053；自己写宿主时要注意这个顺序。

### 6.2 ★ 批处理里别混用 `chcp 65001` 和中文

`.bat` 里同时出现 `chcp 65001` 和中文文本（含注释），cmd 会误解析后续行，
甚至**把 `netsh` 的输出当成命令执行**（报 `'规则名：...' is not recognized`）。
`.bat` 里的 `^` 续行也不可靠。

**规矩**：`.bat` 只做薄封装、**保持纯 ASCII**，逻辑写进 Python（`tools/win_service.py` 的子命令），
中文说明由 Python 打印。防火墙那步就是这么改的。

### 6.3 其他易踩点

- **`sc query` 的退出码 0 不代表服务正常**：从未启动过的服务也是 `STOPPED` + 退出码 0。
- **控制台代码页是 GBK**：输出里带 `⚠`/`✓` 这类字符会让 `print` 抛 `UnicodeEncodeError` 打断命令。
  用 ASCII 标记，并对 stdout 设 `errors="replace"` 兜底。

---

## 7. GPU 训练

### 7.1 代码层面的两个改动点

1. `core/train.py::resolve_device(requested)`
   - `""` / `"auto"` → 有 CUDA 返回 `"0"`，否则 `"cpu"`
   - `"cpu"` → 强制 CPU
   - `"0"` / `"cuda:0"` → 原样透传，无 CUDA 时回落
   **不要在任何地方写死 `device="cpu"`**；GPU 分支才开 `cudnn.benchmark`。
2. 前端 `web/src/views/TrainConfig.vue` 的设备下拉读 `/api/inference/devices`
   （取 `backend === 'pytorch'` 且 `id !== 'cpu'`），默认值 `auto`；
   老缓存里本机不存在的 GPU 编号要回落 `auto`，否则训练一启动就报错。

### 7.2 参数经验值

- 8GB 显存：seg/det/obb `imgsz=640` 用 `batch 8~16`；cls `imgsz=224` 可用 `32~64`。
- CUDA OOM：先 batch 减半，再降模型尺寸（s→n）。
- `workers`：GPU 建议 4~8，纯 CPU 建议 2~8，Windows 异常时回 0。
- cls 任务 `imgsz` 被强制成 224（YOLO11-cls 是 ImageNet 224 预训练，放大到 640 会学不到纹理）。

---

## 8. 数据与存储

### 8.1 SQLite-only 收口检查

- `server/config.py` 的 `DATABASE_URL` 留空 → 自动用 `<STORAGE_ROOT>/app.db`。
- `server/database.py` 只建 SQLite 引擎，并设 `journal_mode=WAL` + `busy_timeout=30000`
  （训练线程写 epoch 日志不会阻塞 API 读）。
- ORM 里不应再有 `celery_task_id` 这类遗留字段。
- 确认没有真依赖：`grep -rn "pymysql\|redis\|celery" server/ core/ tools/`，
  剩下的应该只是"已去除 MySQL/Redis/Celery"这类注释文字。
- **迁移只要拷 `storage/`**（`app.db` + 图片 + runs），不需要导 SQL。

### 8.2 僵尸数据清理

**症状**：前端能列出项目，但缩略图全 404，后端日志刷 `GET /api/images/<id>/file → 404`。
原因：`app.db` 的 `images` 记录指向 `storage/uploads/<project_id>/` 下已不存在的文件
（换机器只拷了 db 没拷 uploads 就会这样）。

```powershell
venv\Scripts\python.exe tools\audit_storage.py                  # 体检，默认只读不改
venv\Scripts\python.exe tools\audit_storage.py --purge-orphan-projects   # 删"文件全丢"的僵尸项目
venv\Scripts\python.exe tools\audit_storage.py --purge-project 3         # 删指定项目
venv\Scripts\python.exe tools\audit_storage.py --purge-missing-images    # 只删丢失的图片记录
venv\Scripts\python.exe tools\audit_storage.py --vacuum                  # VACUUM 回收 app.db 空间
```

清理一个项目 = DB 里删 `projects / defect_classes / images / annotations / train_tasks /
train_epoch_logs / inference_results / exported_models` 的相关行 + 删磁盘
`storage\uploads\<project_id>\` 与 `storage\runs\task_<task_id>\`。
**先停服务再清**（SQLite 写锁 + WAL 会让删除看着"没生效"）。

> 想要干净起点：删掉 `storage\app.db*` 后重启服务即可——`init_db()` 会重建空库和 admin 账号。

---

## 9. 故障速查表

| 现象 | 真因 | 处理 |
|---|---|---|
| `WinError 1114` / c10.dll 初始化失败 | 缺 VC++ Redist | 装 `vc_redist.x64.exe`（≥14.40） |
| `Torch not compiled with CUDA enabled` | 装成 CPU wheel | 重装 `--index-url .../cu126` |
| `no kernel image is available` | wheel 不支持该卡架构 | 降 `--torch-index cu121` |
| 训练日志显示走 CPU | 前端缓存里 device 是 `cpu` | 设备改 `auto`，或清项目训练参数缓存 |
| 首次训练卡在 `Downloading ...` | 离线机器缺预训练权重 | 把 `.pt` 放到项目根 `weights\` |
| `unexpected keyword argument 'workers'` | `server/` 与 `core/` 版本不一致 | 两边一起同步后重启服务 |
| 重启后任务永远"训练中" | 老进程残留状态 | lifespan 会标 failed；不行重启服务 |
| 训练秒失败且 GPU 报 OOM | batch 太大 | batch 减半或模型降档 |
| 提交第二个任务返回 409 | 单并发设计（`TrainingDispatcher`） | 等前一个跑完，属预期 |
| 页面 404 / 空白 | `web/dist` 没构建 | `cd web && npm run build` |
| 本机能开、同事打不开 | 防火墙没放行 | `deploy\open_firewall.bat`（管理员） |
| 服务起不来、日志空白、事件 7009 | 用了 pythonservice.exe 当镜像 | 见 §6.1，重跑 `install_service.bat` |
| `[devices] OpenVINO probe failed` | 没装 OpenVINO（可选组件） | 正常提示；要 OV 推理才装 |

可选组件（不影响训练/标注）：

```powershell
venv\Scripts\python.exe -m pip install openvino onnx onnxslim onnxruntime
```

---

## 10. 验收清单

```powershell
Invoke-RestMethod http://localhost:8000/api/health          # {"status":"ok",...}
# 带 token 后查：/api/inference/devices 应含 CPU + CUDA - <卡名> + total_mem_mb
Get-ChildItem storage\app.db, storage\uploads, storage\runs  # 首次启动后自动生成
```

浏览器：

1. `http://localhost:8000` 能登录、项目列表能打开（前端托管 OK）。
2. 训练页「训练设备」出现 `自动（优先 GPU）` 与显卡项（CUDA 探测 OK）。
3. 提交一个最小训练任务：日志出现 `[train device] 自动选择 GPU: cuda:0（...）`，
   `storage/runs/task_<N>/` 下产出 `results.csv`，`nvidia-smi` 能看到 python 占显存。
4. 局域网另一台电脑能打开 `http://<本机IP>:8000`。

**改完 `server/*` 或 `core/*` 必须重启服务**；`web/src/*` 在 dev 模式下不用重启。
