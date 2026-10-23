<template>
  <div class="page-container" style="max-width:1400px">
    <div class="page-header">
      <div style="display:flex;align-items:center;gap:12px">
        <h1>运行监控</h1>
        <el-tag :type="autoRefresh ? 'success' : 'info'" size="small">
          {{ autoRefresh ? '自动刷新 2s' : '已暂停刷新' }}
        </el-tag>
      </div>
      <div style="display:flex;gap:12px;align-items:center">
        <el-switch v-model="autoRefresh" active-text="自动刷新" />
        <el-button size="small" @click="refreshAll">立即刷新</el-button>
      </div>
    </div>

    <!-- 训练进度 -->
    <el-card shadow="never" style="margin-bottom:16px">
      <template #header>
        <div class="card-head">
          <span>训练进度</span>
          <span class="muted">{{ status.now ? '更新于 ' + fmtTime(status.now) : '' }}</span>
        </div>
      </template>
      <template v-if="activeTask">
        <div class="activity-row">
          <div class="activity-left">
            <el-tag :type="statusType(activeTask.status)" size="small">{{ statusText(activeTask.status) }}</el-tag>
            <span class="activity-name">#{{ activeTask.id }} {{ activeTask.task_name }}</span>
            <el-tag size="small" type="info" effect="plain">{{ activeTask.model_name || '-' }}</el-tag>
          </div>
          <div class="activity-right">
            <span class="muted">epoch {{ activeTask.current_epoch }}/{{ activeTask.epochs }}</span>
          </div>
        </div>
        <el-progress
          v-if="activeTask.status === 'training' || activeTask.status === 'stopping'"
          :percentage="Math.min(100, Math.round(activeTask.pct || 0))"
          :stroke-width="16" :text-inside="true"
          :status="activeTask.status === 'stopping' ? 'warning' : ''"
          style="margin:6px 0 12px"
        />
        <div class="stat-grid">
          <div class="stat"><span class="muted">阶段</span><b>{{ stageText(activeTask.status) }}</b></div>
          <div class="stat"><span class="muted">已用时间</span><b>{{ fmtDuration(activeTask.elapsed_s) }}</b></div>
          <div class="stat"><span class="muted">平均每轮</span><b>{{ fmtDuration(activeTask.avg_epoch_s) }}</b></div>
          <div class="stat"><span class="muted">预计剩余</span><b>{{ fmtDuration(activeTask.eta_s) }}</b></div>
        </div>
        <div class="recent-block">
          <div class="muted" style="margin-bottom:4px">最近输出（实时）</div>
          <div class="recent-lines">
            <div v-for="(l, i) in status.recent_lines" :key="i">{{ l }}</div>
            <div v-if="!status.recent_lines?.length" class="muted">暂无输出</div>
          </div>
        </div>
      </template>
      <div v-else class="idle">
        <el-icon><Coffee /></el-icon>
        <span>当前空闲：没有训练 / 数据集准备 / 导出任务</span>
      </div>
    </el-card>

    <!-- 资源概况 -->
    <el-row :gutter="16" style="margin-bottom:16px">
      <el-col :span="6">
        <el-card shadow="never" class="res-card">
          <div class="res-title">CPU / 内存</div>
          <div class="res-line">CPU <b>{{ fmt(status.cpu_pct, 1) }}%</b> <span class="muted">({{ status.cpu_count || '-' }} 核)</span></div>
          <el-progress :percentage="Math.round(status.cpu_pct || 0)" :show-text="false" :stroke-width="6" />
          <div class="res-line" style="margin-top:8px">
            内存 <b>{{ fmtMb(status.mem?.used_mb) }} / {{ fmtMb(status.mem?.total_mb) }} MB</b>
          </div>
          <el-progress
            :percentage="Math.round(status.mem?.pct || 0)"
            :show-text="false" :stroke-width="6"
            :status="(status.mem?.pct || 0) >= 90 ? 'exception' : ''"
          />
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="never" class="res-card">
          <div class="res-title">磁盘（storage）</div>
          <div class="res-line">剩余 <b>{{ status.disk ? status.disk.free_gb + ' GB' : '-' }}</b></div>
          <div class="res-line muted">总计 {{ status.disk ? status.disk.total_gb + ' GB' : '-' }}</div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="never" class="res-card">
          <div class="res-title">后端进程</div>
          <div class="res-line">PID <b>{{ status.process?.pid ?? '-' }}</b></div>
          <div class="res-line muted">
            运行 {{ fmtUptime(status.process?.uptime_s) }} · 常驻 {{ fmtMb(status.process?.rss_mb) }} MB
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="never" class="res-card">
          <div class="res-title">日志文件</div>
          <div class="res-line">{{ logInfo.file || '未找到日志' }}</div>
          <div class="res-line muted">{{ fmtSize(logInfo.size) }} · {{ logInfo.mtime ? fmtTime(logInfo.mtime) : '-' }}</div>
        </el-card>
      </el-col>
    </el-row>

    <!-- GPU -->
    <el-card v-if="status.gpu?.gpus?.length" shadow="never" style="margin-bottom:16px">
      <template #header>
        <div class="card-head">
          <span>GPU</span>
          <span class="muted">采样于 {{ fmtTime(status.gpu?.sampled_at) }}</span>
        </div>
      </template>
      <div v-for="g in status.gpu.gpus" :key="g.index ?? 0" class="gpu-row">
        <div class="gpu-name">
          <el-tag size="small" type="info">GPU {{ g.index }}</el-tag>
          {{ g.name }}
        </div>
        <div class="gpu-metrics">
          <div><span class="muted">温度</span><b :class="tempClass(g.temperature_c)">{{ fmt(g.temperature_c, 0) }} °C</b></div>
          <div><span class="muted">占用</span><b>{{ fmt(g.utilization_pct, 0) }}%</b></div>
          <div><span class="muted">显存</span><b>{{ fmtMb(g.memory_used_mb) }}/{{ fmtMb(g.memory_total_mb) }} MB</b></div>
          <div><span class="muted">核心/显存频率</span><b>{{ fmt(g.clock_sm_mhz, 0) }} / {{ fmt(g.clock_mem_mhz, 0) }} MHz</b></div>
          <div><span class="muted">功耗</span><b>{{ fmt(g.power_w, 0) }}/{{ fmt(g.power_limit_w, 0) }} W</b></div>
        </div>
      </div>
    </el-card>

    <!-- 实时日志 -->
    <el-card shadow="never">
      <template #header>
        <div class="card-head">
          <span>实时日志</span>
          <div style="display:flex;gap:14px;align-items:center;font-size:12px">
            <el-checkbox v-model="appOnly">仅业务日志</el-checkbox>
            <el-select v-model="lineCount" size="small" style="width:100px">
              <el-option :value="200" label="200 行" />
              <el-option :value="500" label="500 行" />
              <el-option :value="1000" label="1000 行" />
            </el-select>
            <el-checkbox v-model="follow">自动滚到底</el-checkbox>
          </div>
        </div>
      </template>
      <div ref="logBox" class="log-console">
        <div v-for="(l, i) in logs" :key="i" class="log-line" :class="lineClass(l)">{{ l }}</div>
        <div v-if="!logs.length" class="log-empty">暂无日志</div>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount, watch, nextTick, computed } from 'vue'
import api from '../api/index'

interface GpuStat {
  index: number | null
  name: string
  temperature_c: number | null
  utilization_pct: number | null
  memory_used_mb: number | null
  memory_total_mb: number | null
  clock_sm_mhz: number | null
  clock_mem_mhz: number | null
  power_w: number | null
  power_limit_w: number | null
}

interface ActiveTask {
  id: number
  project_id: number
  task_name: string
  status: string
  current_epoch: number
  epochs: number
  model_name: string | null
  started_at: string | null
  elapsed_s: number | null
  avg_epoch_s: number | null
  eta_s: number | null
  pct: number
}

interface LogInfo { file: string | null; size: number; mtime: string | null }

interface StatusPayload {
  now: string
  active_tasks: ActiveTask[]
  dispatcher_task_id: number | null
  cpu_pct: number | null
  cpu_count: number | null
  mem: { used_mb: number; total_mb: number; pct: number } | null
  disk: { free_gb: number; total_gb: number } | null
  process: { pid: number; uptime_s: number; rss_mb: number } | null
  gpu: { available: boolean; sampled_at?: string | null; gpus: GpuStat[] } | null
  log: LogInfo | null
  recent_lines?: string[]
}

const status = ref<StatusPayload>({
  now: '',
  active_tasks: [],
  dispatcher_task_id: null,
  cpu_pct: null,
  cpu_count: null,
  mem: null,
  disk: null,
  process: null,
  gpu: null,
  log: null,
  recent_lines: [],
})
const activeTask = computed<ActiveTask | null>(() => status.value.active_tasks[0] || null)
const logInfo = ref<LogInfo>({ file: null, size: 0, mtime: null })
const logs = ref<string[]>([])
const autoRefresh = ref(true)
const appOnly = ref(true)
const lineCount = ref(200)
const follow = ref(true)
const logBox = ref<HTMLElement | null>(null)
let timer: any = null

async function loadStatus() {
  try {
    const { data } = await api.get<StatusPayload>('/monitor/status')
    status.value = data
  } catch {
    // 忽略瞬时错误，下一轮继续
  }
}

async function loadLogs() {
  try {
    const { data } = await api.get('/monitor/logs', {
      params: { lines: lineCount.value, app_only: appOnly.value },
    })
    logs.value = data.lines || []
    logInfo.value = { file: data.file, size: data.size, mtime: data.mtime }
    if (follow.value) await scrollToBottom()
  } catch {
    // 忽略
  }
}

async function scrollToBottom() {
  await nextTick()
  const el = logBox.value
  if (el) el.scrollTop = el.scrollHeight
}

async function refreshAll() {
  await Promise.all([loadStatus(), loadLogs()])
}

function startTimer() {
  stopTimer()
  timer = setInterval(() => {
    if (autoRefresh.value) refreshAll()
  }, 2000)
}
function stopTimer() {
  if (timer) { clearInterval(timer); timer = null }
}

onMounted(async () => {
  await refreshAll()
  startTimer()
})
onBeforeUnmount(stopTimer)

// 切换过滤/行数时立即刷新一次
watch([appOnly, lineCount], () => loadLogs())

// ---- 格式化 ----
function fmt(v: number | null | undefined, digits = 0) {
  return v == null ? '-' : v.toFixed(digits)
}
function fmtMb(v: number | null | undefined) {
  return v == null ? '-' : Math.round(v).toLocaleString()
}
function fmtSize(bytes: number | null | undefined) {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  let n = bytes, i = 0
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++ }
  return `${n.toFixed(1)} ${units[i]}`
}
function fmtUptime(sec: number | null | undefined) {
  if (sec == null) return '-'
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60)
  return h > 0 ? `${h}h ${m}m` : `${m}m`
}
function fmtTime(iso: string | null | undefined) {
  if (!iso) return '-'
  return iso.replace('T', ' ')
}
function pct(t: ActiveTask) {
  if (!t.epochs) return 0
  return Math.min(100, Math.round((t.current_epoch / t.epochs) * 100))
}
function fmtDuration(sec: number | null | undefined) {
  if (sec == null) return '-'
  const s = Math.max(0, Math.round(sec))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const r = s % 60
  if (h > 0) return `${h}h ${m}m`
  if (m > 0) return `${m}m ${r}s`
  return `${r}s`
}
function stageText(s: string) {
  return ({
    pending: '排队中',
    preparing: '数据集准备（导出/形态学/滑窗/划分）',
    training: '训练 / 验证中',
    stopping: '停止中（等当前 epoch 结束）',
    exporting: '导出模型',
  } as any)[s] || s
}
function statusType(s: string) {
  return ({ pending: 'info', preparing: 'warning', training: '', stopping: 'warning', exporting: 'warning' } as any)[s] || 'info'
}
function statusText(s: string) {
  return ({ pending: '排队中', preparing: '准备数据', training: '训练中', stopping: '停止中', exporting: '导出中' } as any)[s] || s
}
function tempClass(t: number | null | undefined) {
  if (t == null) return ''
  if (t >= 85) return 'hot'
  if (t >= 75) return 'warm'
  return ''
}
function lineClass(line: string) {
  if (/ERROR|Traceback|Exception/i.test(line)) return 'error'
  if (/WARNING|⚠️/.test(line)) return 'warn'
  return ''
}
</script>

<style scoped>
.card-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 600;
}
.muted {
  color: #909399;
  font-size: 12px;
  font-weight: normal;
}
.activity-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  padding: 10px 0;
  border-bottom: 1px dashed #ebeef5;
}
.activity-row:last-child {
  border-bottom: none;
}
.activity-left {
  display: flex;
  align-items: center;
  gap: 8px;
}
.activity-name {
  font-weight: 600;
  font-size: 14px;
}
.activity-right {
  display: flex;
  align-items: center;
  gap: 10px;
}
.stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 10px;
  margin-bottom: 12px;
}
.stat {
  background: #fafbfc;
  border: 1px solid #ebeef5;
  border-radius: 8px;
  padding: 8px 10px;
  display: flex;
  flex-direction: column;
  gap: 2px;
  font-size: 13px;
}
.stat b {
  font-size: 15px;
  color: #303133;
}
.recent-block {
  border-top: 1px dashed #ebeef5;
  padding-top: 8px;
}
.recent-lines {
  background: #1d1e1f;
  color: #d4d4d4;
  font-family: Consolas, 'Courier New', monospace;
  font-size: 12px;
  line-height: 1.55;
  border-radius: 8px;
  padding: 8px 10px;
  max-height: 150px;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-all;
}
.idle {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #909399;
  font-size: 13px;
  padding: 12px 0;
}
.res-card {
  min-height: 118px;
}
.res-title {
  font-weight: 600;
  font-size: 13px;
  margin-bottom: 8px;
}
.res-line {
  font-size: 13px;
  margin-bottom: 4px;
  color: #303133;
}
.gpu-row {
  padding: 8px 0;
}
.gpu-name {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
  font-size: 13px;
  margin-bottom: 8px;
}
.gpu-metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 10px;
  font-size: 13px;
}
.gpu-metrics > div {
  display: flex;
  flex-direction: column;
  gap: 2px;
  background: #fafbfc;
  border: 1px solid #ebeef5;
  border-radius: 8px;
  padding: 8px 10px;
}
.gpu-metrics b.warm { color: #e6a23c; }
.gpu-metrics b.hot { color: #f56c6c; }
.log-console {
  background: #1d1e1f;
  color: #d4d4d4;
  font-family: Consolas, 'Courier New', monospace;
  font-size: 12px;
  line-height: 1.6;
  height: 420px;
  overflow: auto;
  border-radius: 8px;
  padding: 10px 12px;
  white-space: pre-wrap;
  word-break: break-all;
}
.log-line.error { color: #f56c6c; }
.log-line.warn { color: #e6a23c; }
.log-empty { color: #888; }
</style>
