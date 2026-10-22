import api from './index'

export interface TrainTask {
  id: number
  project_id: number
  task_name: string
  status: string
  epochs: number
  current_epoch: number
  best_map50: number | null
  best_model_path: string | null
  output_dir: string | null
  error_message: string | null
  started_at: string | null
  finished_at: string | null
  created_at: string
  config?: Record<string, any> | null   // 训练时使用的完整参数（model_name/epochs/lr0/...）
}

export interface EpochLog {
  epoch: number
  train_box_loss: number | null
  train_seg_loss: number | null
  train_cls_loss: number | null
  train_dfl_loss: number | null
  val_box_loss: number | null
  val_seg_loss: number | null
  val_cls_loss: number | null
  val_dfl_loss: number | null
  precision_b: number | null
  recall_b: number | null
  map50_b: number | null
  map50_95_b: number | null
  map50_m: number | null
  map50_95_m: number | null
  top1_acc: number | null   // cls 任务 top1 准确率
  top5_acc: number | null   // cls 任务 top5 准确率
  lr: number | null
}

export interface GpuStat {
  index: number | null
  name: string
  temperature_c: number | null
  utilization_pct: number | null
  memory_used_mb: number | null
  memory_total_mb: number | null
  memory_pct: number | null
  clock_sm_mhz: number | null
  clock_mem_mhz: number | null
  power_w: number | null
  power_limit_w: number | null
  fan_pct: number | null
}

export interface GpuStatsPayload {
  available: boolean
  reason?: string | null
  sampled_at?: string | null
  gpus: GpuStat[]
}

export const trainApi = {
  listTasks: (projectId: number) =>
    api.get<TrainTask[]>(`/projects/${projectId}/train/tasks`),

  getTask: (taskId: number) =>
    api.get<TrainTask>(`/train/tasks/${taskId}`),

  getEpochLogs: (taskId: number) =>
    api.get<EpochLog[]>(`/train/tasks/${taskId}/epochs`),

  // 训练监控页的显卡实时状态（温度/显存/利用率/频率/功耗）
  getGpuStats: () =>
    api.get<GpuStatsPayload>('/gpu/stats'),

  cancelTask: (taskId: number) =>
    api.post(`/train/tasks/${taskId}/cancel`),

  deleteTask: (taskId: number) =>
    api.delete(`/train/tasks/${taskId}`),
}
