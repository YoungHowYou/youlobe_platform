# Labelyou 离线受控部署说明

适用目标：一台不能访问外网的 Windows 电脑，部署目录固定为 `D:\labelyou`，后端业务代码不以 `.py` 明文交付，普通平台账号除删除项目外可正常使用，模型权重按只读资产管理，并启用机器码绑定。

本平台只依赖本地硬盘：数据库为 SQLite 文件（`storage\app.db`），训练任务在 API 进程内的后台线程执行。目标机**不需要**预装或运行 MySQL / Redis / MariaDB / Celery，训练仅支持 CPU。

## 交付包内容

离线包名称类似：

```text
labelyou_offline_YYYYMMDD_<commit>.zip
```

解压后目录结构：

```text
D:\labelyou\
  app\                 后端业务 pyc-only 包，不包含 server/core 的 .py 明文
  web\dist\            前端构建产物
  venv\                已安装依赖的 Python 环境
  runtime\
    python\            内置 Python 运行时
  storage\             数据、上传图片、训练结果、模型权重（含 app.db）
  config\
    .env               离线部署配置
    license.dat        机器授权文件，需由管理员生成后放入
  logs\
  tools\
    get_machine_code.bat
    labelyou_get_machine_code.py
    start_api.bat
    start_all.bat
    smoke_test.bat
    gpu_check.bat
  README_FIRST.md
```

包内不包含 `.git`、后端 `.py` 源码、开发日志、当前机器 `.env`、历史数据、node_modules 源码工程。

## 部署前准备

目标电脑需要提前确认：

- Windows 10/11 64 位
- D 盘有足够空间，建议至少 30GB 可用
- 安装了 Visual C++ 运行库（PyTorch CPU 版依赖）

目标电脑不能联网，所以不要在目标机上执行 `pip install`、`npm install` 或 Ultralytics 自动下载模型。所有依赖均已内置。

## CPU 训练注意事项

- 本平台训练强制 CPU（`core/train.py` 会忽略任何 GPU 设备参数）
- 建议 `batch=4~16` 起步，按内存余量调整；`workers=2~8`
- seg/obb 大图滑窗训练较慢，必要时降低 `epochs` 或 `crop_size`
- 推理设备仍可按需选择（CPU / OpenVINO CPU 等，取决于打包的依赖）

## 机器码绑定流程

目标机第一次部署时还没有 `license.dat`，后端会因为授权缺失拒绝启动。先采集机器码：

```cmd
D:\labelyou\tools\get_machine_code.bat
```

把输出的 `Labelyou machine code` 发给管理员。

管理员在打包机或管理机上执行：

```cmd
D:\yolo26s_platform\venv\Scripts\python.exe D:\yolo26s_platform\tools\labelyou_make_license.py ^
  --machine-code <目标机机器码> ^
  --customer "公司离线电脑" ^
  --out D:\license.dat
```

私钥默认路径：

```text
D:\labelyou_license_private_key.pem
```

注意：私钥只能由管理员保管，不要复制到目标电脑。

生成后，把 `license.dat` 复制到：

```text
D:\labelyou\config\license.dat
```

后端启动时会校验：

- license 签名是否合法
- license 里的机器码是否等于当前电脑机器码
- 如设置了过期时间，是否仍在有效期内

## 平台账号权限

平台内置两类角色：

- `admin`：管理员账号，由你保管
- `user`：普通账号，给目标电脑使用者

普通账号策略：

- 可以新建项目
- 可以导入项目/标注
- 可以上传图片
- 可以标注、修改标注、删除图片
- 可以训练
- 可以推理
- 可以导出/下载项目数据
- 不可以删除项目
- 不可以删除训练任务，因为会移除训练权重
- 不可以删除导出模型记录，因为会移除模型产物
- 不能进入用户管理，用户管理接口仍由管理员控制

前端会隐藏普通用户无权使用的删除按钮；后端中间件仍会强制拦截，不能只依赖前端。

## 模型权重只读策略

模型权重作为平台资产管理：

- 普通用户可以通过平台选择模型进行推理
- 普通用户可以下载允许下载的模型或导出件
- 普通用户不能删除项目
- 普通用户不能删除训练任务
- 普通用户不能删除导出模型记录

如果需要更强保护，可在目标电脑上把 `D:\labelyou\storage` 设置为只有服务账号可写，普通 Windows 登录用户不直接访问文件夹。

## 安装步骤

1. 以管理员身份登录目标电脑。

2. 解压部署包到：

   ```text
   D:\labelyou
   ```

3. 检查配置文件：

   ```text
   D:\labelyou\config\.env
   ```

   关键配置：

   ```env
   DATABASE_URL=                      # 留空 = 使用 storage\app.db（SQLite）
   STORAGE_ROOT=D:\labelyou\storage
   LABELYOU_LICENSE_REQUIRED=True
   LABELYOU_LICENSE_FILE=D:\labelyou\config\license.dat
   YOLO_AUTOINSTALL=False
   ```

4. 采集机器码，生成并放入 `license.dat`。

5. 启动服务：

   ```cmd
   D:\labelyou\tools\start_all.bat
   ```

   第一次启动后端会自动建表（`storage\app.db`），并创建默认管理员。

6. 打开浏览器：

   ```text
   http://localhost:8000
   ```

7. 管理员登录，创建普通用户账号。

   默认管理员账号仍为：

   ```text
   admin / admin123
   ```

   首次部署后请立刻修改管理员密码。

## 启动脚本

后端 API（含训练线程）：

```cmd
D:\labelyou\tools\start_api.bat
```

一键启动：

```cmd
D:\labelyou\tools\start_all.bat
```

基础冒烟测试：

```cmd
D:\labelyou\tools\smoke_test.bat
```

## 验证清单

部署完成后按顺序验证：

- 后端日志出现 license OK
- 后端日志出现 Application startup complete
- `storage\app.db` 文件已生成
- 浏览器能打开 `http://localhost:8000`
- `admin` 可登录
- 管理员能创建普通用户
- 普通用户能新建项目
- 普通用户看不到“删除项目”按钮
- 普通用户调用删除项目接口返回 403
- 在线推理可以跑一张小图
- 训练任务建议先用小数据集、`batch=4` 试跑，确认能进入 training 状态并产出 epoch 日志

## 常见问题

### 后端启动报 license file not found

还没有把 `license.dat` 放到：

```text
D:\labelyou\config\license.dat
```

先运行 `get_machine_code.bat`，把机器码发给管理员生成授权。

### 后端启动报 license is not valid for this machine

`license.dat` 不是为这台电脑生成的，或者目标机硬件信息变化导致机器码变化。重新采集机器码并签发。

### 后端启动卡在 Ultralytics 自动安装

确认 `.env` 或启动脚本里有：

```env
YOLO_AUTOINSTALL=False
```

### 训练任务一直显示“排队中 / 训练中”

训练在 API 进程内的守护线程执行。若 API 进程被重启，上次未跑完的任务会在下次启动时自动标记为 failed（错误信息“服务重启，训练中断”），重新创建任务即可。同一时间只允许一个训练任务，第二个任务会返回 409。

### 普通用户仍能删除项目

确认后端是新包，且浏览器强制刷新。真正权限在后端中间件，普通用户删除接口应返回 403。
