# 快速启动指南

## 前提条件

- Python 3.9+ 已安装
- Node.js 18+ 已安装

> 本平台只依赖本地硬盘（SQLite 文件库 + 文件存储），**不需要** Docker / MySQL / Redis。

---

## 第一步：解压并进入项目

```bash
unzip yolo26s_platform.zip
cd yolo26s_platform
```

---

## 第二步：启动后端

```bash
# 安装 Python 依赖
pip install -r server/requirements.txt

# 复制环境变量配置
cp .env.example .env

# 启动 FastAPI（首次启动会自动建 SQLite 库 storage/app.db）
uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
```

看到以下输出说明后端启动成功：

```
============================================================
  YOLO26s-Seg 标注训练平台 后端已启动
  API 文档: http://localhost:8000/docs
============================================================
```

**打开浏览器访问 http://localhost:8000/docs** 确认 Swagger 文档能加载。

---

## 第三步：启动前端（新开一个终端）

```bash
cd yolo26s_platform/web
npm install
npm run dev
```

看到以下输出：

```
  VITE v5.x.x  ready in xxx ms
  ➜  Local:   http://localhost:5173/
```

---

## 第四步：打开浏览器使用

访问 **http://localhost:5173**

操作流程：

1. 点击「新建项目」→ 填写名称 → 点击「创建」
2. 点击项目卡片进入详情页
3. 点击「上传图像」→ 拖入你的硅片图像（BMP/PNG/JPG）
4. 点击任意缩略图 → 进入标注编辑器
5. 选择类别 → 点击画多边形 → 双击完成 → Ctrl+S 保存

---

## 快捷键（标注编辑器）

| 快捷键 | 功能 |
|--------|------|
| 鼠标点击 | 添加多边形顶点 |
| 双击 / Enter | 完成当前多边形 |
| Esc | 取消当前绘制 |
| Ctrl+S | 保存标注 |
| Ctrl+Z | 撤销 |
| Ctrl+Y | 重做 |
| A / ← | 上一张图像 |
| D / → | 下一张图像 |
| 滚轮 | 缩放画布 |
| Alt+拖拽 / 右键拖拽 | 平移画布 |
| 双击画布 | 重置视图 |

---

## 常见问题

### Q: 训练任务一直显示"排队中 / 训练中"

训练在 uvicorn 进程内的后台线程执行。若后端被重启，上次未跑完的任务会在下次启动时自动标记为 failed（错误信息"服务重启，训练中断"），重新创建任务即可。同一时间只允许一个训练任务。

### Q: npm install 报错

确保 Node.js 版本 >= 18：

```bash
node -v
```

### Q: 上传图像后缩略图不显示

检查后端终端有没有报错。确认 `storage/uploads/` 目录有写权限。

