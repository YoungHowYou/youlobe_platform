<template>
  <div class="page-container" style="max-width:1600px">
    <div class="page-header">
      <div style="display:flex;align-items:center;gap:12px">
        <el-button text @click="router.push(`/project/${id}`)"><el-icon><ArrowLeft /></el-icon> 返回项目</el-button>
        <h1>训练监控</h1>
      </div>
      <el-button class="hbtn hbtn--blue" @click="router.push(`/project/${id}/train`)">
        <el-icon><Plus /></el-icon> 新建训练
      </el-button>
    </div>

    <!-- 任务列表 -->
    <el-card shadow="never" style="margin-bottom:20px">
      <template #header><span style="font-weight:600">训练任务</span></template>
      <el-table :data="tasks" stripe @row-click="selectTask" highlight-current-row style="width:100%">
        <el-table-column prop="id" label="ID" width="60" />
        <el-table-column prop="task_name" label="任务名称" min-width="160" />
        <el-table-column label="状态" width="190">
          <template #default="{row}">
            <el-tag :type="statusType(row.status)" size="small">{{ statusText(row.status) }}</el-tag>
            <el-tag v-if="row.queue_position" type="info" size="small" effect="plain" style="margin-left:4px">
              队列第 {{ row.queue_position }} 位
            </el-tag>
            <el-tag
              v-if="row.config?.stop_reason === 'manual'"
              type="warning" size="small" effect="plain" style="margin-left:4px"
            >
              早停@{{ row.config?.stopped_at_epoch }}ep
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="进度" width="200">
          <template #default="{row}">
            <el-progress :percentage="row.epochs>0?Math.round(row.current_epoch/row.epochs*100):0" :status="row.status==='completed'?'success':row.status==='failed'?'exception':''" :stroke-width="14" :text-inside="true">
            </el-progress>
            <span style="font-size:11px;color:#999">{{ row.current_epoch }}/{{ row.epochs }}</span>
          </template>
        </el-table-column>
        <el-table-column label="最佳指标" width="150">
          <template #default="{row}">
            <span v-if="row.best_map50" style="font-weight:600;color:#67C23A">
              {{ row.config?.task_type === 'cls' ? 'Top-1' : 'mAP50' }}
              {{ (row.best_map50*100).toFixed(1) }}%
            </span>
            <span v-else style="color:#999">-</span>
          </template>
        </el-table-column>
        <el-table-column label="耗时" width="120">
          <template #default="{row}">
            <span v-if="isActiveStatus(row.status) && row.started_at">{{ duration(row.started_at, null) }}</span>
            <span v-else-if="isActiveStatus(row.status)">等待中...</span>
            <span v-else-if="row.started_at && row.finished_at">{{ duration(row.started_at, row.finished_at) }}</span>
            <span v-else style="color:#999">-</span>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="180">
          <template #default="{row}">
            <el-button
              v-if="row.status==='training' && row.current_epoch>0"
              type="primary" text size="small"
              @click.stop="stopTask(row.id)"
            >早停</el-button>
            <el-button v-if="row.status==='training'||row.status==='preparing'" type="warning" text size="small" @click.stop="cancelTask(row.id)">取消</el-button>
            <!-- 排队中（未开训）的任务可退队，也可以直接删除 -->
            <el-button v-if="row.status==='pending'" type="warning" text size="small" @click.stop="cancelTask(row.id)">退出队列</el-button>
            <el-button v-if="isAdmin && (row.status==='completed'||row.status==='failed'||row.status==='cancelled'||row.status==='pending')" type="danger" text size="small" @click.stop="deleteTask(row.id)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 训练曲线 -->
    <template v-if="selectedTaskId">
      <!-- 错误信息 -->
      <el-alert v-if="selectedTask?.error_message" type="error" :closable="false" style="margin-bottom:16px">
        <template #title>训练失败</template>
        <pre style="white-space:pre-wrap;font-size:12px;max-height:200px;overflow:auto">{{ selectedTask.error_message }}</pre>
      </el-alert>

      <!-- 训练计时 -->
      <el-card v-if="selectedTask" shadow="never" class="time-card">
        <template #header>
          <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-weight:600">训练计时 — {{ selectedTask.task_name }}</span>
            <el-tag :type="statusType(selectedTask.status)" size="small">{{ statusText(selectedTask.status) }}</el-tag>
          </div>
        </template>
        <div class="time-grid">
          <div class="time-item">
            <div class="time-label">已用时间</div>
            <div class="time-value">{{ selectedTiming.elapsed }}</div>
            <div class="time-tip">{{ selectedTiming.startedAt }}</div>
          </div>
          <div class="time-item">
            <div class="time-label">平均每轮</div>
            <div class="time-value">{{ selectedTiming.avgEpoch }}</div>
            <div class="time-tip">按已完成 epoch 估算</div>
          </div>
          <div class="time-item">
            <div class="time-label">预计剩余</div>
            <div class="time-value">{{ selectedTiming.remaining }}</div>
            <div class="time-tip">{{ selectedTiming.progress }}</div>
          </div>
          <div class="time-item">
            <div class="time-label">预计完成</div>
            <div class="time-value">{{ selectedTiming.finishAt }}</div>
            <div class="time-tip">{{ selectedTiming.finishedAt }}</div>
          </div>
        </div>
      </el-card>

      <!-- GPU 实时状态（nvidia-smi 采样，只读传感器，不影响训练） -->
      <el-card v-if="gpuStats.gpus.length" shadow="never" class="gpu-card">
        <template #header>
          <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-weight:600">GPU 状态</span>
            <span style="font-size:12px;color:#909399">
              <template v-if="gpuStats.sampled_at">采样于 {{ gpuStats.sampled_at.replace('T', ' ') }}</template>
              <template v-else>每 3 秒刷新</template>
            </span>
          </div>
        </template>
        <div v-for="g in gpuStats.gpus" :key="g.index ?? 0" class="gpu-row">
          <div class="gpu-name">
            <el-tag size="small" type="info">GPU {{ g.index }}</el-tag>
            <span>{{ g.name }}</span>
          </div>
          <div class="gpu-metrics">
            <div class="gpu-metric">
              <div class="gpu-metric-label">温度</div>
              <div class="gpu-metric-value" :class="gpuTempClass(g.temperature_c)">
                {{ fmtGpuNum(g.temperature_c, 0) }} <span class="unit">°C</span>
              </div>
            </div>
            <div class="gpu-metric">
              <div class="gpu-metric-label">GPU 占用</div>
              <div class="gpu-metric-value">{{ fmtGpuNum(g.utilization_pct, 0) }} <span class="unit">%</span></div>
              <el-progress :percentage="Math.round(g.utilization_pct ?? 0)" :show-text="false" :stroke-width="6" />
            </div>
            <div class="gpu-metric">
              <div class="gpu-metric-label">显存</div>
              <div class="gpu-metric-value">
                {{ fmtGpuMb(g.memory_used_mb) }} <span class="unit">/ {{ fmtGpuMb(g.memory_total_mb) }} MB</span>
              </div>
              <el-progress
                :percentage="Math.min(100, Math.round(g.memory_pct ?? 0))"
                :show-text="false"
                :stroke-width="6"
                :status="(g.memory_pct ?? 0) >= 90 ? 'exception' : ''"
              />
            </div>
            <div class="gpu-metric">
              <div class="gpu-metric-label">核心频率</div>
              <div class="gpu-metric-value">{{ fmtGpuNum(g.clock_sm_mhz, 0) }} <span class="unit">MHz</span></div>
            </div>
            <div class="gpu-metric">
              <div class="gpu-metric-label">显存频率</div>
              <div class="gpu-metric-value">{{ fmtGpuNum(g.clock_mem_mhz, 0) }} <span class="unit">MHz</span></div>
            </div>
            <div class="gpu-metric">
              <div class="gpu-metric-label">功耗</div>
              <div class="gpu-metric-value">
                {{ fmtGpuNum(g.power_w, 0) }} <span class="unit">/ {{ fmtGpuNum(g.power_limit_w, 0) }} W</span>
              </div>
            </div>
          </div>
        </div>
      </el-card>

      <!-- 训练参数（点击展开/收起） -->
      <el-card v-if="selectedTask?.config" shadow="never" style="margin-bottom:20px">
        <template #header>
          <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-weight:600">训练参数 — {{ selectedTask.task_name }}</span>
            <div style="display:flex;gap:8px">
              <el-button text size="small" @click="exportConfig">
                <el-icon><Download /></el-icon> 导出 JSON
              </el-button>
              <el-button text size="small" @click="paramsExpanded = !paramsExpanded">
                {{ paramsExpanded ? '收起' : '展开' }}
              </el-button>
            </div>
          </div>
        </template>
        <div v-show="paramsExpanded">
          <div v-for="group in paramGroups" :key="group.title" class="param-group">
            <div class="param-group-title">{{ group.title }}</div>
            <div class="param-grid">
              <div
                v-for="item in group.items.filter(i => hasValue(i.key))"
                :key="item.key"
                class="param-item"
              >
                <div class="param-label">{{ item.label }}</div>
                <div class="param-value" :title="item.tip || ''">{{ formatValue(item.key, item.fmt) }}</div>
                <div v-if="item.tip" class="param-tip">{{ item.tip }}</div>
              </div>
            </div>
          </div>
        </div>
        <!-- 收起时只显示关键 4 项 -->
        <div v-show="!paramsExpanded" class="param-summary">
          <span class="ps-item"><b>模型:</b> {{ cfg.model_name || '-' }}</span>
          <span class="ps-item"><b>Epochs:</b> {{ cfg.epochs || '-' }}</span>
          <span class="ps-item"><b>BatchSize:</b> {{ cfg.batch_size || '-' }}</span>
          <span class="ps-item"><b>Workers:</b> {{ formatValue('workers') }}</span>
          <span class="ps-item"><b>lr0:</b> {{ cfg.lr0 ?? '-' }}</span>
          <span class="ps-item"><b>degrees:</b> {{ cfg.degrees ?? '-' }}</span>
          <span class="ps-item"><b>device:</b> {{ cfg.device ?? '-' }}</span>
        </div>
      </el-card>

      <!-- Loss 曲线 -->
      <el-card shadow="never" style="margin-bottom:20px">
        <template #header>
          <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-weight:600">Loss 曲线 — {{ selectedTask?.task_name }}</span>
            <span style="font-size:12px;color:#999">Epoch {{ epochs.length }} | 蓝=Train 红=Val</span>
          </div>
        </template>
        <!-- cls：单一 loss，宽图展示 -->
        <div v-if="isClsTask" class="chart-box">
          <div ref="chartClsLoss" style="width:100%;height:320px"></div>
        </div>
        <!-- seg/det/obb：按实际存在的 loss 项动态渲染（YOLO26 没有 dfl） -->
        <div v-else class="charts-grid">
          <div v-for="def in lossDefs" :key="def.key" class="chart-box">
            <div :ref="(el: any) => setLossChartEl(def.key, el)" style="width:100%;height:280px"></div>
          </div>
        </div>
      </el-card>

      <!-- 验证指标曲线 -->
      <el-card shadow="never" style="margin-bottom:20px">
        <template #header>
          <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-weight:600">验证指标</span>
            <span style="font-size:12px;color:#999">Ultralytics 每轮只在验证集上评估，因此本区只有验证集曲线</span>
          </div>
        </template>
        <!-- cls：Top1 / Top5 准确率 -->
        <div v-if="isClsTask" class="charts-grid">
          <div class="chart-box">
            <div ref="chartTop1" style="width:100%;height:280px"></div>
          </div>
          <div class="chart-box">
            <div ref="chartTop5" style="width:100%;height:280px"></div>
          </div>
        </div>
        <!-- seg/det/obb：Precision / Recall / mAP@50 / mAP@50:95 -->
        <div v-else class="charts-grid">
          <div class="chart-box">
            <div ref="chartPrecision" style="width:100%;height:280px"></div>
          </div>
          <div class="chart-box">
            <div ref="chartRecall" style="width:100%;height:280px"></div>
          </div>
          <div class="chart-box">
            <div ref="chartMap50" style="width:100%;height:280px"></div>
          </div>
          <div class="chart-box">
            <div ref="chartMap5095" style="width:100%;height:280px"></div>
          </div>
        </div>
      </el-card>

      <!-- 学习率曲线 -->
      <el-card shadow="never" style="margin-bottom:20px">
        <template #header><span style="font-weight:600">学习率</span></template>
        <div ref="chartLr" style="width:100%;height:250px"></div>
      </el-card>
    </template>

    <el-empty v-else-if="tasks.length===0" description="暂无训练任务" />
    <el-empty v-else description="点击上方任务查看训练曲线" />
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount, watch, nextTick, computed } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import * as echarts from 'echarts'
import { trainApi, type TrainTask, type EpochLog, type GpuStatsPayload } from '../api/train'

const props = defineProps<{ id: string }>()
const router = useRouter()
const isAdmin = computed(() => {
  try {
    return JSON.parse(localStorage.getItem('user') || '{}')?.role === 'admin'
  } catch {
    return false
  }
})
const projectId = parseInt(props.id)

const tasks = ref<TrainTask[]>([])
const selectedTaskId = ref<number|null>(null)
const epochs = ref<EpochLog[]>([])
let pollTimer: any = null
let clockTimer: any = null
const nowTs = ref(Date.now())

// ---- GPU 实时状态（温度/显存/占用/频率/功耗）----
const gpuStats = ref<GpuStatsPayload>({ available: false, gpus: [] })
let gpuTimer: any = null

async function loadGpuStats() {
  try {
    const { data } = await trainApi.getGpuStats()
    gpuStats.value = data
  } catch {
    // 后端尚未重启/接口不存在时静默失败，不影响监控页其他内容
  }
}

function fmtGpuNum(v: number | null | undefined, digits = 0) {
  return v == null ? '-' : v.toFixed(digits)
}
function fmtGpuMb(v: number | null | undefined) {
  return v == null ? '-' : Math.round(v).toLocaleString()
}
function gpuTempClass(t: number | null | undefined) {
  if (t == null) return ''
  if (t >= 85) return 'hot'
  if (t >= 75) return 'warm'
  return ''
}

const selectedTask = computed(() => tasks.value.find(t => t.id === selectedTaskId.value))

// 是否分类任务（cls 的指标和 seg/det/obb 不同：单一 loss + Top1/Top5 准确率）
const isClsTask = computed(() => (selectedTask.value?.config as any)?.task_type === 'cls')
const selectedTiming = computed(() => getTiming(selectedTask.value || null))

// ========== 训练参数展示 ==========
const paramsExpanded = ref(true)
const cfg = computed<Record<string, any>>(() => (selectedTask.value?.config || {}) as any)

interface ParamItem { key: string; label: string; tip?: string; fmt?: 'pct' | 'fix2' | 'fix4' | 'json' }
interface ParamGroup { title: string; items: ParamItem[] }

const paramGroups: ParamGroup[] = [
  {
    title: '基本参数',
    items: [
      { key: 'task_type',     label: '任务类型',  tip: 'seg=分割 / det=检测 / cls=分类 / obb=旋转检测' },
      { key: 'model_name',    label: '模型',      tip: '权重文件名（如 yolo11s-obb.pt）' },
      { key: 'epochs',        label: 'Epochs',    tip: '训练总轮数' },
      { key: 'batch_size',    label: 'Batch Size',tip: '一次梯度更新的图片数' },
      { key: 'patience',      label: 'Patience',  tip: '早停容忍轮数（连续无提升）' },
      { key: 'device',        label: '设备',      tip: 'GPU 优先；无可用 CUDA 时自动回落 CPU' },
      { key: 'workers',       label: 'DataLoader Workers', tip: '数据加载/增强子进程数；Windows 异常时可设为 0' },
      { key: 'train_mode',    label: '训练模式',  tip: 'scratch=从头 / finetune=继承' },
      { key: 'resume_from_task_id', label: '继承自', tip: '继承训练时来源任务 ID' },
      { key: 'train_ratio',   label: '训练集占比' },
    ],
  },
  {
    title: '学习率',
    items: [
      { key: 'lr0',             label: '初始学习率 lr0' },
      { key: 'lrf',             label: '终末学习率 lrf', tip: '相对 lr0 的倍数；最终 lr = lr0 × lrf' },
      { key: 'momentum',        label: 'Momentum' },
      { key: 'weight_decay',    label: 'Weight Decay' },
      { key: 'warmup_epochs',   label: 'Warmup Epochs' },
      { key: 'warmup_momentum', label: 'Warmup Momentum' },
    ],
  },
  {
    title: '颜色增广',
    items: [
      { key: 'hsv_h', label: 'HSV-H (色调)' },
      { key: 'hsv_s', label: 'HSV-S (饱和度)' },
      { key: 'hsv_v', label: 'HSV-V (亮度)' },
    ],
  },
  {
    title: '几何增广',
    items: [
      { key: 'degrees',   label: '旋转角度 (°)', tip: 'OBB 任务通常 180，HBB 任务建议 ≤ 30' },
      { key: 'translate', label: '平移比例' },
      { key: 'scale',     label: '缩放范围' },
      { key: 'shear',     label: '剪切角度 (°)' },
      { key: 'flipud',    label: '上下翻转概率' },
      { key: 'fliplr',    label: '左右翻转概率' },
    ],
  },
  {
    title: '高级增广',
    items: [
      { key: 'mosaic',       label: 'Mosaic',      tip: '4 图拼接增强概率' },
      { key: 'mixup',        label: 'Mixup',       tip: '图像混叠概率' },
      { key: 'copy_paste',   label: 'Copy-Paste',  tip: '仅 seg/obb 有效' },
      { key: 'erasing',      label: 'Random Erasing' },
      { key: 'close_mosaic', label: '末尾关闭 Mosaic 轮数' },
    ],
  },
  {
    title: '形态学预处理（仅 seg）',
    items: [
      { key: 'use_morphology',     label: '启用形态学' },
      { key: 'dilate_kernel',      label: '膨胀核' },
      { key: 'erode_kernel',       label: '腐蚀核' },
      { key: 'mask_dilate_kernel', label: 'Mask 膨胀核' },
    ],
  },
  {
    title: '数据集预处理',
    items: [
      { key: 'resize_h',          label: 'Resize H' },
      { key: 'resize_w',          label: 'Resize W' },
      { key: 'crop_size',         label: '切割尺寸' },
      { key: 'overlap',           label: '滑窗重叠率' },
      { key: 'oversample_factor', label: '过采样倍数' },
    ],
  },
]

function hasValue(k: string): boolean {
  if (k === 'workers') return true
  const v = cfg.value[k]
  return v !== undefined && v !== null && v !== ''
}
function formatValue(k: string, fmt?: string): string {
  const v = cfg.value[k]
  if (k === 'workers' && (v === undefined || v === null || v === '')) return '未记录（旧任务）'
  if (v === undefined || v === null) return '-'
  if (typeof v === 'boolean') return v ? '是' : '否'
  if (Array.isArray(v)) return JSON.stringify(v)
  if (typeof v === 'object') return JSON.stringify(v)
  if (fmt === 'pct' && typeof v === 'number') return (v * 100).toFixed(2) + '%'
  if (fmt === 'fix2' && typeof v === 'number') return v.toFixed(2)
  if (fmt === 'fix4' && typeof v === 'number') return v.toFixed(4)
  return String(v)
}
function exportConfig() {
  const name = selectedTask.value?.task_name || 'train-config'
  const blob = new Blob([JSON.stringify(cfg.value, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  const safe = String(name).replace(/[\\/:*?"<>|\s]+/g, '_')
  a.download = `train-params-${safe}.json`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
  ElMessage.success('训练参数已导出为 JSON 文件，可在训练配置页「导入参数」')
}

// Chart refs
// Loss 图按实际数据动态生成（见 lossDefs），用 key → DOM 的映射存句柄
const lossChartEls: Record<string, HTMLElement> = {}
function setLossChartEl(key: string, el: any) {
  if (el) lossChartEls[key] = el as HTMLElement
}
const LOSS_DEFS = [
  { key: 'box', title: 'Box Loss' },
  { key: 'seg', title: 'Seg Loss' },
  { key: 'cls', title: 'Cls Loss' },
  { key: 'dfl', title: 'DFL Loss' },
]
// 只保留库里真有数据的 loss 项（YOLO26 没有 dfl_loss，就不再画空图）
const lossDefs = computed(() => {
  const ep = epochs.value as any[]
  if (!ep.length) return LOSS_DEFS
  const shown = LOSS_DEFS.filter(d =>
    ep.some(e => e[`train_${d.key}_loss`] != null || e[`val_${d.key}_loss`] != null))
  return shown.length ? shown : LOSS_DEFS
})
const chartClsLoss = ref<HTMLElement>()
const chartPrecision = ref<HTMLElement>()
const chartRecall = ref<HTMLElement>()
const chartMap50 = ref<HTMLElement>()
const chartMap5095 = ref<HTMLElement>()
// cls 专用
const chartTop1 = ref<HTMLElement>()
const chartTop5 = ref<HTMLElement>()
const chartLr = ref<HTMLElement>()

let charts: echarts.ECharts[] = []

// 抽出来以便在 onBeforeUnmount 里 removeEventListener
function onWindowResize() {
  charts.forEach(c => c.resize())
}

onMounted(async () => {
  // 监听 resize 必须在 mount 时注册，并且在 unmount 时移除——
  // 否则每访问一次本页面就会泄漏一个监听器，
  // 引用已被 dispose 的旧 charts 数组，下次窗口缩放时报错
  window.addEventListener('resize', onWindowResize)

  await loadTasks()
  // 自动选中最新任务
  if (tasks.value.length > 0) {
    selectTask(tasks.value[0])
  }
  // GPU 状态：与训练无关，始终刷新（3 秒）
  await loadGpuStats()
  gpuTimer = setInterval(loadGpuStats, 3000)
  // 轮询刷新（训练中时每 5 秒刷新）
  pollTimer = setInterval(async () => {
    const running = tasks.value.some(t => t.status === 'training' || t.status === 'preparing' || t.status === 'pending')
    if (running) {
      await loadTasks()
      if (selectedTaskId.value) await loadEpochs()
    }
  }, 5000)
  clockTimer = setInterval(() => { nowTs.value = Date.now() }, 1000)
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', onWindowResize)
  if (pollTimer) clearInterval(pollTimer)
  if (clockTimer) clearInterval(clockTimer)
  if (gpuTimer) clearInterval(gpuTimer)
  charts.forEach(c => c.dispose())
  charts = []
})

async function loadTasks() {
  const { data } = await trainApi.listTasks(projectId)
  tasks.value = data
}

async function selectTask(row: TrainTask) {
  selectedTaskId.value = row.id
  await loadEpochs()
}

async function loadEpochs() {
  if (!selectedTaskId.value) return
  const { data } = await trainApi.getEpochLogs(selectedTaskId.value)
  epochs.value = data
  await nextTick()
  renderCharts()
}

async function cancelTask(taskId: number) {
  const row = tasks.value.find(t => t.id === taskId)
  const queued = row?.status === 'pending'
  try {
    if (queued) {
      await ElMessageBox.confirm(
        '该任务还在队列中（未开始训练），确定退出队列？',
        '退出队列',
        { type: 'warning', confirmButtonText: '退出队列', cancelButtonText: '保留' }
      )
    } else {
      await ElMessageBox.confirm(
        '将请求训练在最近的检查点友好停止，已产生的 best.pt / last.pt 会尽量保留。确定取消？',
        '友好取消训练',
        { type: 'warning', confirmButtonText: '请求取消', cancelButtonText: '继续训练' }
      )
    }
    await trainApi.cancelTask(taskId)
    ElMessage.success(queued ? '已从训练队列移除' : '已请求友好取消，训练会在最近检查点停止')
    loadTasks()
  } catch {}
}

// 手动早停：当前 epoch 结束后停止，产物按“已完成”保留（不是取消）
async function stopTask(taskId: number) {
  try {
    await ElMessageBox.confirm(
      '早停会在当前 epoch 结束后停止训练，并保留 best.pt / last.pt（状态记为“已完成”，可直接导出/继承）。\n注意：最坏需要等一个 epoch。确定早停？',
      '手动早停',
      { type: 'warning', confirmButtonText: '早停', cancelButtonText: '继续训练' }
    )
    await trainApi.stopTask(taskId)
    ElMessage.success('已请求早停，将在当前 epoch 结束后停止')
    loadTasks()
  } catch {}
}

async function deleteTask(taskId: number) {
  try {
    await ElMessageBox.confirm('确定删除此训练任务？', '确认', { type: 'warning' })
    await trainApi.deleteTask(taskId)
    ElMessage.success('已删除')
    if (selectedTaskId.value === taskId) selectedTaskId.value = null
    loadTasks()
  } catch {}
}

// ====== ECharts ======
function makeChart(el: HTMLElement | undefined): echarts.ECharts | null {
  if (!el) return null
  let c = echarts.getInstanceByDom(el)
  if (!c) {
    c = echarts.init(el)
    charts.push(c)
  }
  return c
}

function renderCharts() {
  const ep = epochs.value
  const x = ep.map(e => e.epoch + 1)

  if (isClsTask.value) {
    // ---- 分类任务：Cls Loss + Top1 / Top5 ----
    // 后端已把 train/loss、val/loss 别名落到 train_cls_loss / val_cls_loss
    renderLossChart(chartClsLoss.value, 'Classification Loss', x,
      ep.map(e => e.train_cls_loss), ep.map(e => e.val_cls_loss))
    renderMetricChart(chartTop1.value, 'Top-1 Accuracy', x, ep.map(e => e.top1_acc), '#67C23A')
    renderMetricChart(chartTop5.value, 'Top-5 Accuracy', x, ep.map(e => e.top5_acc), '#409EFF')
  } else {
    // ---- 检测/分割/旋转：Loss 按实际数据动态出图 + 4 张 Metric ----
    const raw = ep as any[]
    for (const def of lossDefs.value) {
      renderLossChart(lossChartEls[def.key], def.title, x,
        raw.map(e => e[`train_${def.key}_loss`]), raw.map(e => e[`val_${def.key}_loss`]))
    }

    renderMetricChart(chartPrecision.value, 'Precision', x, ep.map(e => e.precision_b), '#409EFF')
    renderMetricChart(chartRecall.value, 'Recall', x, ep.map(e => e.recall_b), '#E6A23C')
    renderMetricChart(chartMap50.value, 'mAP@0.5 (Box)', x, ep.map(e => e.map50_b), '#67C23A')
    renderMetricChart(chartMap5095.value, 'mAP@0.5:0.95 (Box)', x, ep.map(e => e.map50_95_b), '#F56C6C')
  }

  // LR chart 通用
  renderMetricChart(chartLr.value, 'Learning Rate', x, ep.map(e => e.lr), '#909399')
}

function renderLossChart(el: HTMLElement | undefined, title: string, x: number[], train: (number|null)[], val: (number|null)[]) {
  const chart = makeChart(el)
  if (!chart) return
  const hasTrain = train.some(v => v != null)
  const hasVal = val.some(v => v != null)
  const series: any[] = []
  if (hasTrain) {
    series.push({ name: 'Train', type: 'line', data: train, smooth: true, lineStyle: { width: 2 }, itemStyle: { color: '#409EFF' }, symbol: 'none' })
  }
  if (hasVal) {
    series.push({ name: 'Val', type: 'line', data: val, smooth: true, lineStyle: { width: 2, type: 'dashed' }, itemStyle: { color: '#F56C6C' }, symbol: 'none' })
  }
  chart.setOption({
    title: {
      text: title,
      subtext: hasTrain ? '' : '无训练集数据（历史任务未记录，可用 tools/backfill_epoch_logs.py 回填）',
      subtextStyle: { color: '#E6A23C', fontSize: 11 },
      left: 'center', top: 0, textStyle: { fontSize: 14 },
    },
    tooltip: { trigger: 'axis' },
    legend: { data: series.map(s => s.name), bottom: 0 },
    grid: { left: 50, right: 20, top: 46, bottom: 40 },
    xAxis: { type: 'category', data: x, name: 'Epoch' },
    yAxis: { type: 'value', name: 'Loss' },
    series,
  }, true)
}

function renderMetricChart(el: HTMLElement | undefined, title: string, x: number[], data: (number|null)[], color: string) {
  const chart = makeChart(el)
  if (!chart) return

  // 找最大值
  const valid = data.filter(v => v !== null && v !== 0) as number[]
  const maxVal = valid.length > 0 ? Math.max(...valid) : 0
  const maxIdx = data.indexOf(maxVal)

  chart.setOption({
    title: { text: title, left: 'center', top: 0, textStyle: { fontSize: 14 } },
    tooltip: { trigger: 'axis', formatter: (p: any) => {
      const v = p[0]
      return `Epoch ${v.name}<br/>${v.seriesName}: <b>${v.value !== null ? Number(v.value).toFixed(5) : '-'}</b>`
    }},
    grid: { left: 60, right: 20, top: 40, bottom: 24 },
    xAxis: { type: 'category', data: x, name: 'Epoch' },
    yAxis: { type: 'value' },
    series: [{
      name: title, type: 'line', data, smooth: true,
      lineStyle: { width: 2, color },
      itemStyle: { color },
      symbol: 'none',
      areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
        { offset: 0, color: color + '33' },
        { offset: 1, color: color + '05' },
      ])},
      markPoint: maxVal > 0 ? {
        data: [{ type: 'max', name: 'Best' }],
        label: { formatter: (p: any) => Number(p.value).toFixed(4) },
        symbolSize: 40,
      } : undefined,
    }],
  }, true)
}

// Utils
function statusType(s: string) {
  return ({ pending: 'info', preparing: 'warning', training: '', stopping: 'warning', completed: 'success', failed: 'danger', cancelled: 'info' } as any)[s] || 'info'
}
function statusText(s: string) {
  return ({ pending: '排队中', preparing: '准备数据', training: '训练中', stopping: '停止中', completed: '已完成', failed: '失败', cancelled: '已取消' } as any)[s] || s
}
function isActiveStatus(s: string) {
  return ['pending', 'preparing', 'training', 'stopping', 'exporting'].includes(s)
}
function duration(start: string, end: string | null) {
  const s = new Date(start).getTime()
  const e = end ? new Date(end).getTime() : nowTs.value
  return formatSeconds(Math.max(0, Math.round((e - s) / 1000)))
}
function formatSeconds(sec: number) {
  if (sec < 60) return `${sec}秒`
  if (sec < 3600) return `${Math.floor(sec/60)}分${sec%60}秒`
  const h = Math.floor(sec / 3600)
  const m = Math.floor((sec % 3600) / 60)
  return `${h}时${m}分`
}
function formatDateTime(ts: number | string | null) {
  if (!ts) return '-'
  const d = new Date(ts)
  if (Number.isNaN(d.getTime())) return '-'
  return d.toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}
function getTiming(task: TrainTask | null) {
  if (!task || !task.started_at) {
    return {
      elapsed: '-',
      avgEpoch: '-',
      remaining: '-',
      finishAt: '-',
      startedAt: '尚未开始',
      finishedAt: '',
      progress: '-',
    }
  }

  const start = new Date(task.started_at).getTime()
  const end = task.finished_at ? new Date(task.finished_at).getTime() : nowTs.value
  const elapsedSec = Math.max(0, Math.round((end - start) / 1000))
  const totalEpochs = Math.max(0, task.epochs || 0)
  const doneEpochs = Math.max(0, Math.min(task.current_epoch || epochs.value.length || 0, totalEpochs || task.current_epoch || 0))
  const avgSec = doneEpochs > 0 ? elapsedSec / doneEpochs : 0
  const remainingEpochs = Math.max(0, totalEpochs - doneEpochs)
  const remainingSec = avgSec > 0 && isActiveStatus(task.status) ? Math.round(avgSec * remainingEpochs) : 0

  return {
    elapsed: formatSeconds(elapsedSec),
    avgEpoch: avgSec > 0 ? formatSeconds(Math.round(avgSec)) : '-',
    remaining: isActiveStatus(task.status) && avgSec > 0 ? formatSeconds(remainingSec) : (task.status === 'completed' ? '已完成' : '-'),
    finishAt: isActiveStatus(task.status) && avgSec > 0 ? formatDateTime(nowTs.value + remainingSec * 1000) : (task.finished_at ? formatDateTime(task.finished_at) : '-'),
    startedAt: `开始：${formatDateTime(task.started_at)}`,
    finishedAt: task.finished_at ? `结束：${formatDateTime(task.finished_at)}` : (avgSec > 0 ? '基于当前平均速度' : '等待首轮完成后估算'),
    progress: totalEpochs > 0 ? `${doneEpochs}/${totalEpochs} epoch` : `${doneEpochs} epoch`,
  }
}

// 窗口 resize 监听器在 onMounted / onBeforeUnmount 配对管理（见上方）
</script>

<style scoped>
.charts-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
.chart-box {
  border: 1px solid #ebeef5;
  border-radius: 8px;
  padding: 8px;
}
pre {
  margin: 0;
  font-family: Consolas, monospace;
}

.time-card {
  margin-bottom: 20px;
}
.time-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 10px;
}
.time-item {
  background: #fafbfc;
  border: 1px solid #ebeef5;
  border-radius: 6px;
  padding: 10px 12px;
  min-width: 0;
}
.time-label {
  font-size: 12px;
  color: #909399;
  margin-bottom: 4px;
}
.time-value {
  font-size: 18px;
  font-weight: 700;
  color: #303133;
  line-height: 1.25;
  word-break: break-all;
}
.time-tip {
  font-size: 11px;
  color: #c0c4cc;
  margin-top: 4px;
  min-height: 16px;
}

/* 训练参数展示 */
.param-group {
  margin-bottom: 18px;
}
.param-group-title {
  font-size: 13px;
  font-weight: 600;
  color: #303133;
  border-left: 3px solid #409EFF;
  padding-left: 8px;
  margin-bottom: 10px;
}
.param-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
  gap: 8px;
}
.param-item {
  background: #fafbfc;
  border: 1px solid #ebeef5;
  border-radius: 6px;
  padding: 8px 10px;
  min-width: 0;
}
.param-label {
  font-size: 11px;
  color: #909399;
  margin-bottom: 2px;
}
.param-value {
  font-size: 13px;
  font-weight: 600;
  color: #303133;
  word-break: break-all;
}
.param-tip {
  font-size: 10px;
  color: #c0c4cc;
  margin-top: 2px;
  line-height: 1.3;
}
.param-summary {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  font-size: 13px;
  color: #606266;
}
.ps-item b {
  color: #909399;
  font-weight: normal;
  margin-right: 4px;
}

/* ---- GPU 状态卡片 ---- */
.gpu-card {
  margin-bottom: 20px;
}
.gpu-row {
  padding: 10px 0;
  border-bottom: 1px dashed #ebeef5;
}
.gpu-row:last-child {
  border-bottom: none;
}
.gpu-name {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  font-weight: 600;
  color: #303133;
  margin-bottom: 10px;
}
.gpu-metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
}
.gpu-metric {
  background: #fafbfc;
  border: 1px solid #ebeef5;
  border-radius: 8px;
  padding: 8px 10px;
}
.gpu-metric-label {
  font-size: 12px;
  color: #909399;
  margin-bottom: 4px;
}
.gpu-metric-value {
  font-size: 16px;
  font-weight: 600;
  color: #303133;
  margin-bottom: 6px;
}
.gpu-metric-value .unit {
  font-size: 12px;
  font-weight: normal;
  color: #909399;
}
.gpu-metric-value.warm {
  color: #e6a23c;
}
.gpu-metric-value.hot {
  color: #f56c6c;
}
</style>
