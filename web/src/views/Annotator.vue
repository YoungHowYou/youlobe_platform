<template>
  <div class="annotator" tabindex="0" ref="rootRef">
    <div class="toolbar">
      <div class="toolbar-left">
        <el-button text @click="goBack"><el-icon><ArrowLeft /></el-icon> 返回</el-button>
        <el-divider direction="vertical" />
        <span class="filename">{{ currentImage?.filename }}</span>
        <el-tag :type="statusTagType" size="small" style="margin-left:8px">{{ statusLabel }}</el-tag>
      </div>
      <div class="toolbar-center">
        <el-radio-group v-model="currentTool" size="small" @change="onToolChange">
          <el-radio-button label="select">选择</el-radio-button>
          <el-radio-button label="polygon">多边形</el-radio-button>
          <el-radio-button label="rect">矩形</el-radio-button>
          <el-radio-button label="brush">涂抹</el-radio-button>
          <el-radio-button label="circle">圆形</el-radio-button>
          <el-radio-button label="polyline">折线</el-radio-button>
          <el-radio-button label="eraser">擦除</el-radio-button>
          <el-radio-button label="boxEraser">框选擦除</el-radio-button>
        </el-radio-group>
        <el-divider direction="vertical" />
        <div v-if="showBrushSize" class="brush-size-ctrl">
          <span class="ctrl-label">{{ currentTool==='eraser'?'橡皮':'画笔' }}</span>
          <el-slider v-model="brushSize" :min="1" :max="500" :step="1" style="width:100px" />
          <el-input-number v-model="brushSize" :min="1" :max="500" :step="1" :controls="false" size="small" style="width:64px" />
          <span class="ctrl-value">px</span>
        </div>
        <template v-if="currentTool==='circle'">
          <el-divider direction="vertical" />
          <el-radio-group v-model="circleMode" size="small" @change="cancelDrawing">
            <el-radio-button label="center">圆心+半径</el-radio-button>
            <el-radio-button label="three">3点共圆</el-radio-button>
          </el-radio-group>
        </template>
        <el-divider direction="vertical" />
        <el-button size="small" :disabled="!canUndo" @click="undo">撤销</el-button>
        <el-button size="small" :disabled="!canRedo" @click="redo">重做</el-button>
        <template v-if="selectedAnnIdx>=0">
          <el-divider direction="vertical" />
          <span class="ctrl-label">选中区域</span>
          <el-button size="small" title="逆时针旋转 15°（快捷键 [ ，加 Shift 为 1°）" @click="rotateSelectedBy(-15)">-15°</el-button>
          <el-button size="small" title="顺时针旋转 15°（快捷键 ] ，加 Shift 为 1°）" @click="rotateSelectedBy(15)">+15°</el-button>
          <el-button size="small" title="顺时针旋转 90°" @click="rotateSelectedBy(90)">+90°</el-button>
        </template>
        <el-button size="small" @click="zoomFit">适应窗口</el-button>
        <el-button size="small" title="向左旋转90°（仅旋转视图，保存坐标自动换算回原图）" @click="rotateView(-90)"><el-icon><RefreshLeft /></el-icon></el-button>
        <el-button size="small" title="向右旋转90°（快捷键 R）" @click="rotateView(90)"><el-icon><RefreshRight /></el-icon></el-button>
        <el-button size="small" type="danger" :disabled="annotations.length===0" @click="clearAll">清空</el-button>
      </div>
      <div class="toolbar-right">
        <el-select
          v-model="inferModelIdx"
          size="small"
          class="infer-model-select"
          :disabled="inferModels.length===0"
          placeholder="选择推理模型"
        >
          <el-option v-for="(m,i) in inferModels" :key="i" :label="m.label" :value="i" />
        </el-select>
        <el-button
          size="small"
          type="primary"
          :loading="inferLoading"
          :disabled="inferModels.length===0 || !currentImage"
          @click="runInferCurrent"
        >
          推理标注
        </el-button>
        <span class="save-indicator" :class="saveState">{{ saveStateText }}</span>
        <el-button v-if="currentImage?.status!=='reviewed'" @click="markAsOk" size="small" type="success">标记OK</el-button>
        <el-button v-else @click="unmarkOk" size="small" type="warning">取消OK</el-button>
      </div>
    </div>
    <div class="main-area">
      <div class="nav-panel">
        <div class="nav-header">图像列表</div>
        <div class="nav-list">
          <div v-for="img in imageList" :key="img.id" :class="['nav-item',{active:img.id===currentImageId},imgSC(img)]" @click="switchImage(img.id)">
            <img :src="getThumbUrl(img.id)" class="nav-thumb" loading="lazy" />
            <div class="nav-info">
              <div class="nav-filename">{{ img.filename }}</div>
              <div class="nav-status-row">
                <span :class="['status-dot','dot-'+img.status]"></span>
                <span class="nav-status-text">{{ stText(img.status) }}</span>
                <span v-if="img.annotation_count>0" class="nav-ann-count">{{ img.annotation_count }}</span>
              </div>
            </div>
          </div>
        </div>
        <div class="nav-footer">
          <el-button size="small" :disabled="!hasPrev" @click="prevImage">◀</el-button>
          <el-button size="small" :disabled="!hasNext" @click="nextImage">▶</el-button>
        </div>
      </div>
      <div class="canvas-area" ref="canvasAreaRef">
        <canvas ref="canvasEl" id="annotationCanvas"></canvas>
        <div v-if="toolHint" class="draw-hint">{{ toolHint }}</div>
      </div>
      <div class="side-panel">
        <div class="panel-section">
          <div class="panel-title">当前类别</div>
          <div class="class-list">
            <div v-for="dc in defectClasses" :key="dc.id" :class="['class-item',{active:selectedClassId===dc.id}]" @click="selectedClassId=dc.id!">
              <span class="class-dot" :style="{background:dc.color}"></span>
              <span class="class-name">{{ dc.name }}</span>
              <span class="class-count">{{ cntCls(dc.id!) }}</span>
            </div>
          </div>
        </div>
        <div class="panel-section" style="flex:1;overflow-y:auto">
          <div class="panel-title">标注列表 ({{ annotations.length }})</div>
          <div v-if="annotations.length===0" class="empty-hint">暂无标注</div>
          <div v-for="(ann,idx) in annotations" :key="idx" :class="['ann-item',{selected:selectedAnnIdx===idx}]" @click="selectAnn(idx)">
            <span class="class-dot" :style="{background:clsColor(ann.class_id)}"></span>
            <!-- 已画好的标注可直接改类别（换色 + 计入对应类别统计，可撤销） -->
            <el-select
              :model-value="ann.class_id"
              size="small"
              class="ann-class-select"
              :title="clsName(ann.class_id)"
              @change="(v:number)=>changeAnnClass(idx,v)"
              @click.stop
            >
              <el-option v-for="dc in defectClasses" :key="dc.id" :label="dc.name" :value="dc.id!" />
            </el-select>
            <span class="ann-points">{{ ann.polygon.length }}点</span>
            <el-button type="danger" text size="small" @click.stop="delAnn(idx)"><el-icon><Delete /></el-icon></el-button>
          </div>
        </div>
      </div>
    </div>
    <div class="statusbar">
      <span v-if="currentImage">{{ currentImage.width }}×{{ currentImage.height }}</span>
      <span>标注:{{ annotations.length }}</span>
      <span>缩放:{{ Math.round(zoomLevel*100) }}%</span>
      <span v-if="viewRot!==0" style="color:#E6A23C">视图旋转:{{ viewRot }}°</span>
      <span v-if="currentImageIdx>=0">{{ currentImageIdx+1 }}/{{ imageList.length }}</span>
      <span class="status-shortcuts">空格+拖拽移动 · Alt+拖拽区域平移 · 选中后 Shift+方向键微调 · 滚轮缩放</span>
      <div v-if="isSegTask" class="seg-view-controls">
        <el-switch
          v-model="showAnnotationLayer"
          size="small"
          inline-prompt
          active-text="显"
          inactive-text="隐"
        />
        <el-radio-group v-model="imageChannel" size="small" class="channel-radio">
          <el-radio-button label="O">O</el-radio-button>
          <el-radio-button label="R">R</el-radio-button>
          <el-radio-button label="G">G</el-radio-button>
          <el-radio-button label="B">B</el-radio-button>
        </el-radio-group>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// @ts-nocheck — fabric 画布文件动态类型多（fabric 无 .d.ts），跳过严格类型检查，不影响运行
import { ref, onMounted, onBeforeUnmount, computed, nextTick, watch } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { fabric } from 'fabric'
import { projectApi, type DefectClass } from '../api/project'
import { imageApi, type ImageInfo } from '../api/image'
import { annotationApi, type AnnotationData, type Point, type AnnShape, type ShapeParams, type InferenceModelInfo } from '../api/annotation'

const props = defineProps<{ projectId: string; imageId: string }>()
const router = useRouter()
const rootRef = ref<HTMLElement>()
const currentImageId = ref(parseInt(props.imageId))
const currentImage = ref<ImageInfo|null>(null)
const imageList = ref<ImageInfo[]>([])
const defectClasses = ref<DefectClass[]>([])
const annotations = ref<AnnotationData[]>([])
const projectTaskType = ref<'seg'|'det'|'cls'|'obb'>('seg')
const selectedClassId = ref<number>(0)
const selectedAnnIdx = ref(-1)
// 拖动/旋转过程中的实时提示（角度、位移），显示在画布底部
const dragHint = ref('')
const currentTool = ref('select')
const brushSize = ref(2)
const showAnnotationLayer = ref(true)
const imageChannel = ref<'O'|'R'|'G'|'B'>('O')
const inferModels = ref<InferenceModelInfo[]>([])
const inferModelIdx = ref(0)
const inferLoading = ref(false)
const selectedInferModel = computed(() => inferModels.value[inferModelIdx.value] || null)
// 切工具 / 切类别 / 切图 时重置 Shift 追加锚点——避免追加到不相关的旧标注
watch(currentTool,()=>{lastBrushAnnIdx=-1;lastBrushPts=[]})
watch(selectedClassId,()=>{lastBrushAnnIdx=-1;lastBrushPts=[]})
watch(showAnnotationLayer,(visible)=>{
  if(suppressViewControlWatch)return
  if(!isSegTask.value)return
  if(!visible){
    selectedAnnIdx.value=-1
    removeVertexHandles()
    cancelDrawing()
  }
  renderAnnotations()
})
watch(imageChannel,()=>{
  if(suppressViewControlWatch)return
  if(!isSegTask.value||!currentImage.value)return
  loadCanvasBackground(currentImageId.value,false)
})
const isDrawing = ref(false)
/** 圆形工具的画法：center=圆心+半径，three=三点共圆 */
const circleMode = ref<'center'|'three'>('center')
const circleClicks = ref(0)   // 仅用于模板提示（circlePts 是普通变量，不响应式）
const saving = ref(false)
const saveState = ref<'idle'|'pending'|'saving'|'saved'|'error'>('idle')
const saveStateText = computed(()=>({
  idle:'',pending:'未保存',saving:'保存中…',saved:'已自动保存',error:'保存失败'
}[saveState.value]))
const zoomLevel = ref(1)
const dirty = ref(false)
const spaceDown = ref(false)
const undoStack = ref<string[]>([])
const redoStack = ref<string[]>([])
const canUndo = computed(()=>undoStack.value.length>0)
const canRedo = computed(()=>redoStack.value.length>0)
const currentImageIdx = computed(()=>imageList.value.findIndex(i=>i.id===currentImageId.value))
const hasPrev = computed(()=>currentImageIdx.value>0)
const hasNext = computed(()=>currentImageIdx.value<imageList.value.length-1)
const statusLabel = computed(()=>({unlabeled:'未标注',labeling:'标注中',labeled:'已标注',reviewed:'OK'})[currentImage.value?.status||'']||'')
const statusTagType = computed(()=>({unlabeled:'info',labeling:'warning',labeled:'',reviewed:'success'})[currentImage.value?.status||''] as any||'info')
const isSegTask = computed(()=>projectTaskType.value==='seg')
const showBrushSize = computed(()=>['brush','circle','polyline','eraser'].includes(currentTool.value))
const toolHint = computed(()=>{
  if(dragHint.value) return dragHint.value
  if(spaceDown.value) return '按住空格拖拽移动画布'
  return({
    polygon:isDrawing.value?'点击加顶点·双击/回车完成·Esc取消':'点击开始画多边形（Alt+拖拽可平移已有区域）',
    rect:'点击拖拽画矩形（Alt+拖拽可平移已有区域）',
    brush:'按住涂抹·松开生成标注·按住 Shift 涂抹追加到上一段（大缺陷分多笔涂）',
    circle:circleMode.value==='three'
      ?`三点共圆：点 3 个点自动生成圆${circleClicks.value?`（已点 ${circleClicks.value}/3）`:''}·Esc 取消`
      :'点击圆心后拖拽出半径（Alt+拖拽可平移已有区域）',
    polyline:isDrawing.value?'点击加折点·双击/回车完成':'点击开始画折线（Alt+拖拽可平移已有区域）',
    eraser:'按住涂抹擦除标注区域',boxEraser:'拖拽框选删除',
    select:'拖动区域平移·拖绿色圆点旋转(Shift 吸附15°)·白点改顶点·矩形拖角/圆拖圆心与半径·Shift+方向键微调·[ ] 旋转15°',
  })[currentTool.value]||''
})
const canvasAreaRef = ref<HTMLElement>()
let canvas:fabric.Canvas|null=null
let drawingPoints:{x:number;y:number}[]=[]
let brushPts:{x:number;y:number}[]=[]
let isBrushing=false
// Shift 追加：分多笔涂同一个缺陷时，按住 Shift 涂的笔迹会自动合并到上一段
// 而非新建独立 annotation。lastBrushAnnIdx 跟踪上次 brush 落到 annotations
// 数组的位置，lastBrushPts 累积笔迹点序列重新 rasterContour 拟合。
let lastBrushAnnIdx=-1
let lastBrushPts:{x:number;y:number}[]=[]
let isErasing=false
let eraserPts:{x:number;y:number}[]=[]
let circleStart:{x:number;y:number}|null=null
// 3 点共圆：已点击的点（显示坐标像素），满 3 个就生成圆
let circlePts:{x:number;y:number}[]=[]
let rectStart:{x:number;y:number}|null=null
let boxEraserStart:{x:number;y:number}|null=null
let editHandles:fabric.Circle[]=[]
let clipboard:AnnotationData|null=null   // 始终存原图坐标系（不随视图旋转失效）
let autoSaveTimer:any=null
let isPanning=false,lastPanX=0,lastPanY=0
let bgLoadSeq=0
let suppressViewControlWatch=false
let hintTimer:any=null
// Alt+拖拽平移：任意工具下都能直接平移区域（不必先切「选择」工具）
let manualDrag:{idx:number;startX:number;startY:number;origPoly:Point[];moved:boolean}|null=null

// ================================================================
// 视图旋转：只旋转显示与编辑坐标系，图片文件和 DB 里的标注始终是原图坐标。
// 内存中 annotations 在"显示坐标系"下工作（这样所有绘图工具零改动），
// 加载时 原图→显示，保存时 显示→原图。仅支持 90° 整数倍，换算无损。
// ================================================================
const viewRot=ref(0)                      // 0|90|180|270，顺时针
const rotMap=new Map<number,number>()     // 本次会话内记住每张图的旋转角

function rotPoint(p:Point,rot:number):Point{
  switch(((rot%360)+360)%360){
    case 90:return{x:1-p.y,y:p.x}
    case 180:return{x:1-p.x,y:1-p.y}
    case 270:return{x:p.y,y:1-p.x}
    default:return{x:p.x,y:p.y}
  }
}
const origToDisp=(p:Point)=>rotPoint(p,viewRot.value)
const dispToOrig=(p:Point)=>rotPoint(p,(360-viewRot.value)%360)
function transformAnns(anns:AnnotationData[],fn:(p:Point)=>Point):AnnotationData[]{
  return anns.map(a=>({...a,polygon:a.polygon.map(fn)}))
}
// 显示坐标系下的图像尺寸（旋转 90/270 时宽高互换）
function dispW():number{if(!currentImage.value)return 1;return viewRot.value%180===0?currentImage.value.width:currentImage.value.height}
function dispH():number{if(!currentImage.value)return 1;return viewRot.value%180===0?currentImage.value.height:currentImage.value.width}

// fabric 以对象左上角为原点旋转；旋转后平移，使图像 bbox 左上角回到场景 (0,0)
function placeBgImage(img:fabric.Image){
  const iw=img.width||1,ih=img.height||1
  let left=0,top=0
  if(viewRot.value===90)left=ih
  else if(viewRot.value===180){left=iw;top=ih}
  else if(viewRot.value===270)top=iw
  img.set({angle:viewRot.value,left,top})
  img.setCoords()
}

function makeChannelElement(src:HTMLImageElement,mode:'R'|'G'|'B',done:(el:HTMLImageElement)=>void){
  const w=src.naturalWidth||src.width
  const h=src.naturalHeight||src.height
  const cv=document.createElement('canvas')
  cv.width=w;cv.height=h
  const ctx=cv.getContext('2d')
  if(!ctx){done(src);return}
  try{
    ctx.drawImage(src,0,0,w,h)
    const imgData=ctx.getImageData(0,0,w,h)
    const offset=mode==='R'?0:mode==='G'?1:2
    for(let i=0;i<imgData.data.length;i+=4){
      const v=imgData.data[i+offset]
      imgData.data[i]=v
      imgData.data[i+1]=v
      imgData.data[i+2]=v
    }
    ctx.putImageData(imgData,0,0)
    const out=new Image()
    out.onload=()=>done(out)
    out.src=cv.toDataURL('image/png')
  }catch{
    done(src)
  }
}

function loadCanvasBackground(imageId:number,fit=true){
  if(!canvas)return
  const reqSeq=++bgLoadSeq
  const reqImageId=imageId
  const mode=imageChannel.value
  const url=imageApi.getFileUrl(reqImageId,false)
  const src=new Image()
  src.crossOrigin='anonymous'
  src.onload=()=>{
    if(!canvas||reqSeq!==bgLoadSeq||reqImageId!==currentImageId.value)return
    const addImage=(el:HTMLImageElement)=>{
      if(!canvas||reqSeq!==bgLoadSeq||reqImageId!==currentImageId.value)return
      canvas.getObjects().filter(o=>(o as any)._bg).forEach(o=>canvas!.remove(o))
      const img=new fabric.Image(el,{selectable:false,evented:false,originX:'left',originY:'top'})
      ;(img as any)._bg=true
      placeBgImage(img)
      canvas.add(img)
      canvas.sendToBack(img)
      if(fit)zoomFit()
      else canvas.requestRenderAll()
    }
    if(mode==='O')addImage(src)
    else makeChannelElement(src,mode,addImage)
  }
  src.src=url
}

function rotateView(delta:number){
  if(!canvas||!currentImage.value)return
  cancelDrawing()
  removeVertexHandles()
  const oldRot=viewRot.value
  const newRot=(((oldRot+delta)%360)+360)%360
  // 标注：旧显示坐标系 → 原图坐标系 → 新显示坐标系
  const remap=(p:Point)=>rotPoint(rotPoint(p,(360-oldRot)%360),newRot)
  annotations.value=transformAnns(annotations.value,remap)
  // undo/redo 栈同步换算，避免旋转后撤销跳回旧坐标系
  undoStack.value=undoStack.value.map(s=>JSON.stringify(transformAnns(JSON.parse(s),remap)))
  redoStack.value=redoStack.value.map(s=>JSON.stringify(transformAnns(JSON.parse(s),remap)))
  viewRot.value=newRot
  rotMap.set(currentImageId.value,newRot)
  const bg=canvas.getObjects().find((o:any)=>o._bg) as fabric.Image|undefined
  if(bg)placeBgImage(bg)
  renderAnnotations()
  if(currentTool.value==='select'&&selectedAnnIdx.value>=0)showVertexHandles(selectedAnnIdx.value)
  zoomFit()
}

// ================================================================
// 栅格化→轮廓提取（涂抹/折线共用）
// ================================================================
function rasterContour(points:{x:number;y:number}[],lw:number,W:number,H:number):Point[]{
  if(points.length<2)return[]
  const pad=lw+2
  let x1=Infinity,y1=Infinity,x2=-Infinity,y2=-Infinity
  for(const p of points){if(p.x<x1)x1=p.x;if(p.y<y1)y1=p.y;if(p.x>x2)x2=p.x;if(p.y>y2)y2=p.y}
  x1=Math.max(0,Math.floor(x1-pad));y1=Math.max(0,Math.floor(y1-pad))
  x2=Math.min(W,Math.ceil(x2+pad));y2=Math.min(H,Math.ceil(y2+pad))
  const cw=x2-x1,ch=y2-y1;if(cw<1||ch<1)return[]
  const cv=document.createElement('canvas');cv.width=cw;cv.height=ch
  const ctx=cv.getContext('2d')!;ctx.fillStyle='#000';ctx.fillRect(0,0,cw,ch)
  ctx.strokeStyle='#FFF';ctx.fillStyle='#FFF';ctx.lineWidth=lw;ctx.lineCap='round';ctx.lineJoin='round'
  ctx.beginPath();ctx.moveTo(points[0].x-x1,points[0].y-y1)
  for(let i=1;i<points.length;i++)ctx.lineTo(points[i].x-x1,points[i].y-y1)
  ctx.stroke()
  for(const p of points){ctx.beginPath();ctx.arc(p.x-x1,p.y-y1,lw/2,0,Math.PI*2);ctx.fill()}
  return scanContour(ctx,cw,ch,x1,y1,W,H)
}

function scanContour(ctx:CanvasRenderingContext2D,cw:number,ch:number,ox:number,oy:number,W:number,H:number):Point[]{
  // Moore-Neighbor 边界追踪：沿真实轮廓 8 邻接行走，正确处理 U/S/L/C 形涂抹
  // 旧扫描线算法假设每行白像素连续，会把 S 形等的内凹"水平凸包化"，产生填实 bug
  const img=ctx.getImageData(0,0,cw,ch).data
  const m=new Uint8Array(cw*ch)
  for(let i=0;i<cw*ch;i++)if(img[i*4]>128)m[i]=1
  // 起点：row-major 第一个白点（其 N/NW/NE/W 方向必为黑，可直接从 NE 开始顺时针扫）
  let sx=-1,sy=-1
  for(let y=0;y<ch&&sx<0;y++)for(let x=0;x<cw;x++)if(m[y*cw+x]){sx=x;sy=y;break}
  if(sx<0)return[]
  // 方向编号：0:E 1:SE 2:S 3:SW 4:W 5:NW 6:N 7:NE
  const dxs=[1,1,0,-1,-1,-1,0,1],dys=[0,1,1,1,0,-1,-1,-1]
  const path:{x:number;y:number}[]=[{x:sx,y:sy}]
  let cx=sx,cy=sy,prev=6  // 起点 N 邻居为黑（row-major 第一个白点），从 prev+1=NE 开始查
  const maxIter=cw*ch*4
  for(let it=0;it<maxIter;it++){
    let nd=-1
    for(let i=1;i<=8;i++){
      const d_=(prev+i)%8
      const nx=cx+dxs[d_],ny=cy+dys[d_]
      if(nx<0||nx>=cw||ny<0||ny>=ch)continue
      if(m[ny*cw+nx]){nd=d_;cx=nx;cy=ny;break}
    }
    if(nd<0)break  // 孤立像素
    if(cx===sx&&cy===sy)break  // 回到起点
    path.push({x:cx,y:cy})
    prev=(nd+4)%8  // 新的"上一个黑邻居"位于来时方向的反方向
  }
  if(path.length<3)return[]
  // 等距抽稀到 ≤120 点（一像素一点会爆）
  let pts=path
  if(path.length>120){
    const stp=path.length/120
    pts=[]
    for(let i=0;i<120;i++)pts.push(path[Math.floor(i*stp)])
  }
  return pts.map(p=>({x:(p.x+ox)/W,y:(p.y+oy)/H}))
}

// ================================================================
// 像素级擦除：栅格化标注→减去擦除笔画→重新提取轮廓
// ================================================================
function subtractEraserFromAnnotations(eraserPath:{x:number;y:number}[],eraserWidth:number){
  if(!currentImage.value||eraserPath.length<2)return
  const W=dispW(),H=dispH()
  const pad=eraserWidth+4
  // 擦除路径包围盒
  let ex1=Infinity,ey1=Infinity,ex2=-Infinity,ey2=-Infinity
  for(const p of eraserPath){if(p.x<ex1)ex1=p.x;if(p.y<ey1)ey1=p.y;if(p.x>ex2)ex2=p.x;if(p.y>ey2)ey2=p.y}
  ex1-=pad;ey1-=pad;ex2+=pad;ey2+=pad

  let changed=false
  const newAnns:AnnotationData[]=[]

  for(const ann of annotations.value){
    // 检查标注是否与擦除区域有交集
    let overlap=false
    for(const p of ann.polygon){
      const px=p.x*W,py=p.y*H
      if(px>=ex1&&px<=ex2&&py>=ey1&&py<=ey2){overlap=true;break}
    }
    if(!overlap){newAnns.push(ann);continue}

    // 该标注与擦除区域有交集，需要处理
    // 计算标注的包围盒
    let ax1=Infinity,ay1=Infinity,ax2=-Infinity,ay2=-Infinity
    for(const p of ann.polygon){
      const px=p.x*W,py=p.y*H
      if(px<ax1)ax1=px;if(py<ay1)ay1=py;if(px>ax2)ax2=px;if(py>ay2)ay2=py
    }
    ax1=Math.max(0,Math.floor(ax1-2));ay1=Math.max(0,Math.floor(ay1-2))
    ax2=Math.min(W,Math.ceil(ax2+2));ay2=Math.min(H,Math.ceil(ay2+2))
    const cw=ax2-ax1,ch=ay2-ay1
    if(cw<1||ch<1)continue

    // 创建画布：画标注多边形(白色)，再用擦除笔画减去(黑色)
    const cv=document.createElement('canvas');cv.width=cw;cv.height=ch
    const ctx=cv.getContext('2d')!
    ctx.fillStyle='#000';ctx.fillRect(0,0,cw,ch)

    // 画标注多边形
    ctx.fillStyle='#FFF';ctx.beginPath()
    const pts=ann.polygon.map(p=>({x:p.x*W-ax1,y:p.y*H-ay1}))
    ctx.moveTo(pts[0].x,pts[0].y)
    for(let i=1;i<pts.length;i++)ctx.lineTo(pts[i].x,pts[i].y)
    ctx.closePath();ctx.fill()

    // 用擦除笔画减去（黑色覆盖）
    ctx.globalCompositeOperation='destination-out'
    ctx.strokeStyle='#FFF';ctx.fillStyle='#FFF'
    ctx.lineWidth=eraserWidth;ctx.lineCap='round';ctx.lineJoin='round'
    ctx.beginPath()
    ctx.moveTo(eraserPath[0].x-ax1,eraserPath[0].y-ay1)
    for(let i=1;i<eraserPath.length;i++)ctx.lineTo(eraserPath[i].x-ax1,eraserPath[i].y-ay1)
    ctx.stroke()
    for(const p of eraserPath){ctx.beginPath();ctx.arc(p.x-ax1,p.y-ay1,eraserWidth/2,0,Math.PI*2);ctx.fill()}
    ctx.globalCompositeOperation='source-over'

    // 重新提取轮廓
    const newPoly=scanContour(ctx,cw,ch,ax1,ay1,W,H)
    if(newPoly.length>=3){
      newAnns.push({class_id:ann.class_id,polygon:newPoly})
    }
    // 如果 newPoly 为空，说明标注被完全擦掉了，不加入 newAnns
    changed=true
  }

  if(changed){
    pushUndo()
    annotations.value=newAnns
    dirty.value=true
    renderAnnotations()
    scheduleAutoSave()
  }
}

// ================================================================
// 初始化
// ================================================================
let resizeDebounce:any=null
function onWindowResize(){
  clearTimeout(resizeDebounce)
  resizeDebounce=setTimeout(()=>{
    if(canvas&&canvasAreaRef.value){
      canvas.setWidth(canvasAreaRef.value.clientWidth)
      canvas.setHeight(canvasAreaRef.value.clientHeight)
      zoomFit()
    }
  },200)
}

onMounted(async()=>{
  await loadProject();await loadInferModels();await loadImageList();initCanvas();await loadImageAndAnnotations()
  window.addEventListener('keydown',onKD);window.addEventListener('keyup',onKU)
  window.addEventListener('beforeunload',onBeforeUnload)
  window.addEventListener('resize',onWindowResize)
  rootRef.value?.focus()
})
onBeforeUnmount(async ()=>{
  // 组件卸载前 flush 未保存改动
  if(dirty.value){try{await handleSave({silent:true,immediate:true})}catch{}}
  if(autoSaveTimer)clearTimeout(autoSaveTimer)
  if(resizeDebounce)clearTimeout(resizeDebounce)
  window.removeEventListener('keydown',onKD);window.removeEventListener('keyup',onKU)
  window.removeEventListener('beforeunload',onBeforeUnload)
  window.removeEventListener('resize',onWindowResize)
  if(canvas)canvas.dispose()
})

function onBeforeUnload(e:BeforeUnloadEvent){
  if(dirty.value){
    e.preventDefault()
    e.returnValue=''
    // 同步发起保存（async 不会等，但浏览器 beacon 可以）
    try{
      const url=`/api/images/${currentImageId.value}/annotations`
      // beacon 同样要换算回原图坐标系
      const blob=new Blob([JSON.stringify({annotations:transformAnns(annotations.value,dispToOrig)})],{type:'application/json'})
      navigator.sendBeacon&&navigator.sendBeacon(url,blob)
    }catch{}
  }
}

async function loadProject(){
  const{data}=await projectApi.get(parseInt(props.projectId))
  projectTaskType.value=(data.task_type||'seg') as 'seg'|'det'|'cls'|'obb'
  defectClasses.value=data.defect_classes
  if(data.defect_classes.length>0)selectedClassId.value=data.defect_classes[0].id!
}
async function loadImageList(){
  const{data}=await imageApi.list(parseInt(props.projectId),{page:1,page_size:999})
  imageList.value=data.items
}

async function loadInferModels(){
  try{
    const{data}=await annotationApi.listInferenceModels(parseInt(props.projectId))
    inferModels.value=data.filter(m=>m.model_format==='pytorch'&&m.model_path?.endsWith('.pt'))
    inferModelIdx.value=0
  }catch{
    inferModels.value=[]
  }
}

async function runInferCurrent(){
  if(!currentImage.value)return
  if(!selectedInferModel.value){
    ElMessage.warning('当前项目没有可用的 best.pt 模型')
    return
  }
  inferLoading.value=true
  try{
    const m=selectedInferModel.value
    const{data}=await annotationApi.inferCurrent({
      image_id:currentImageId.value,
      project_id:parseInt(props.projectId),
      task_id:m.task_id,
      model_path:m.model_path,
    })
    if(!data.annotations.length){
      ElMessage.info(data.skipped?`没有可用结果，跳过 ${data.skipped} 个未匹配类别`:'未检出缺陷')
      return
    }
    const inferred=data.annotations.map(a=>({
      class_id:a.class_id,
      polygon:sanitizePolygon(a.polygon).map(origToDisp),
      bbox:a.bbox,
      area:a.area,
    }))
    if(!(await ensureCanAddDefect()))return

    let mode:'replace'|'append'=annotations.value.length>0?'append':'replace'
    if(annotations.value.length>0){
      try{
        await ElMessageBox.confirm(
          `推理得到 ${inferred.length} 个标注。请选择写入方式。`,
          '推理标注',
          {
            confirmButtonText:'替换当前标注',
            cancelButtonText:'追加到当前标注',
            distinguishCancelAndClose:true,
            type:'warning',
          },
        )
        mode='replace'
      }catch(action:any){
        if(action==='cancel')mode='append'
        else return
      }
    }

    pushUndo()
    annotations.value=mode==='replace'?inferred:[...annotations.value,...inferred]
    setLocalImageStatus('labeled',annotations.value.length)
    dirty.value=true
    selectedAnnIdx.value=-1
    removeVertexHandles()
    renderAnnotations()
    scheduleAutoSave()
    ElMessage.success(`已生成 ${inferred.length} 个推理标注${data.skipped?`，跳过 ${data.skipped} 个`:''}`)
  }catch(e:any){
    ElMessage.error(e?.response?.data?.detail||'推理标注失败')
  }finally{
    inferLoading.value=false
  }
}

function initCanvas(){
  if(!canvasAreaRef.value)return
  const w=canvasAreaRef.value.clientWidth,h=canvasAreaRef.value.clientHeight
  canvas=new fabric.Canvas('annotationCanvas',{width:w,height:h,backgroundColor:'#2a2a2a',selection:false})

  canvas.on('mouse:wheel',(opt)=>{
    let z=canvas!.getZoom()*(opt.e.deltaY>0?0.92:1.08)
    z=Math.max(0.05,Math.min(30,z))
    canvas!.zoomToPoint({x:opt.e.offsetX,y:opt.e.offsetY} as fabric.Point,z)
    zoomLevel.value=z;opt.e.preventDefault();opt.e.stopPropagation()
  })

  canvas.on('mouse:down',(opt)=>{
    if(spaceDown.value||opt.e.button===2||opt.e.button===1){
      isPanning=true;lastPanX=opt.e.clientX;lastPanY=opt.e.clientY;canvas!.setCursor('grab');return
    }
    if(opt.e.button!==0)return
    if(isSegTask.value&&!showAnnotationLayer.value)return
    const p=canvas!.getPointer(opt.e),tool=currentTool.value
    // select 工具：点 polygon 选中（顶点 handle 由 fabric 自己处理 evented）
    if(tool==='select'){
      const tgt=opt.target as any
      if(tgt&&tgt._ann)selectAnn(tgt._idx)
      else if(tgt&&tgt._handle){/* 让 fabric 处理拖动 */}
      else if(!tgt){selectedAnnIdx.value=-1;removeVertexHandles();highlightSelected()}
      return
    }
    // Alt+拖拽：任意绘图工具下直接平移鼠标所在区域（不必先切到「选择」工具）。
    // 必须在各绘图分支之前拦截，否则会在区域内部落下新顶点 / 新笔迹。
    if(opt.e.altKey){
      const t=opt.target as any
      if(t&&t._ann){opt.e.preventDefault?.();startManualDrag(t._idx,p);return}
    }
    if(tool==='polygon')onPolyClick(p)
    else if(tool==='polyline')onPolylineClick(p)
    else if(tool==='brush'){isBrushing=true;brushPts=[{x:p.x,y:p.y}]}
    else if(tool==='eraser'){isErasing=true;eraserPts=[{x:p.x,y:p.y}]}
    else if(tool==='circle'){
      // 3 点共圆：点击累加，满 3 个自动成圆；圆心+半径：按下记录圆心，松开拖出半径
      if(circleMode.value==='three')onCircle3Click(p)
      else circleStart={x:p.x,y:p.y}
    }
    else if(tool==='rect')rectStart={x:p.x,y:p.y}
    else if(tool==='boxEraser')boxEraserStart={x:p.x,y:p.y}
  })

  canvas.on('mouse:move',(opt)=>{
    if(isPanning){
      const vpt=canvas!.viewportTransform!
      vpt[4]+=opt.e.clientX-lastPanX;vpt[5]+=opt.e.clientY-lastPanY
      // 必须走 setViewportTransform：它会刷新所有对象的 lineCoords（命中判定用的
      // 视口坐标），直接改 vpt[4]/vpt[5] 会让标注区域平移后彻底点不中、拖不动
      canvas!.setViewportTransform(vpt)
      canvas!.requestRenderAll();lastPanX=opt.e.clientX;lastPanY=opt.e.clientY;return
    }
    const p=canvas!.getPointer(opt.e)
    if(manualDrag){applyManualDrag(p);return}
    if(currentTool.value==='polygon'&&isDrawing.value)drawTmpLine(p)
    else if(currentTool.value==='polyline'&&isDrawing.value)drawTmpLine(p)
    else if(currentTool.value==='brush'&&isBrushing){brushPts.push({x:p.x,y:p.y});drawDotAt(p,clsColor(selectedClassId.value)+'66')}
    else if(currentTool.value==='eraser'&&isErasing){eraserPts.push({x:p.x,y:p.y});drawDotAt(p,'rgba(255,80,80,0.5)')}
    else if(currentTool.value==='circle'&&circleStart)drawCirclePrev(p)
    else if(currentTool.value==='circle'&&circleMode.value==='three'&&circlePts.length===2)drawCirclePrev3(p)
    else if(currentTool.value==='rect'&&rectStart)drawRectPrev(p)
    else if(currentTool.value==='boxEraser'&&boxEraserStart)drawBoxRect(p)
  })

  canvas.on('mouse:up',(opt)=>{
    if(isPanning){isPanning=false;canvas!.setCursor('default');return}
    if(manualDrag){finishManualDrag();return}
    const p=canvas!.getPointer(opt.e)
    if(currentTool.value==='brush'&&isBrushing)finishBrush(!!(opt.e as any).shiftKey)
    else if(currentTool.value==='eraser'&&isErasing)finishEraser()
    else if(currentTool.value==='circle'&&circleStart)finishCircle(p)
    else if(currentTool.value==='rect'&&rectStart)finishRect(p)
    else if(currentTool.value==='boxEraser'&&boxEraserStart)finishBoxEraser()
  })

  canvas.on('mouse:dblclick',()=>{
    if(currentTool.value==='polygon'&&isDrawing.value)finishPolygon()
    else if(currentTool.value==='polyline'&&isDrawing.value)finishPolyline()
  })
  canvasAreaRef.value.addEventListener('contextmenu',e=>e.preventDefault())
}

function onToolChange(){
  cancelDrawing()
  // 重建 polygon 让 selectable 状态切换（select 工具可拖整体，其他工具不可）
  renderAnnotations()
  // 仅在 select 工具下显示顶点 handle，其他工具关掉避免事件冲突
  if(currentTool.value==='select'){
    if(selectedAnnIdx.value>=0) showVertexHandles(selectedAnnIdx.value)
  }else{
    removeVertexHandles()
  }
}

async function loadImageAndAnnotations(){
  if(!canvas)return
  currentImage.value=imageList.value.find(i=>i.id===currentImageId.value)||null
  viewRot.value=rotMap.get(currentImageId.value)??0
  suppressViewControlWatch=true
  showAnnotationLayer.value=true
  imageChannel.value='O'
  suppressViewControlWatch=false
  removeVertexHandles()
  canvas.clear();canvas.backgroundColor='#2a2a2a'
  // 锁定本次请求对应的 imageId；如果用户在 await 期间切了图，丢弃这次返回，
  // 否则旧请求的标注会把新图覆盖（实际表现：切到新图却显示前一张的标注）
  const reqImageId=currentImageId.value
  loadCanvasBackground(reqImageId,true)
  try{
    const{data}=await annotationApi.get(reqImageId)
    if(reqImageId!==currentImageId.value)return  // 切图了，本次响应作废
    annotations.value=data.map(a=>({
      class_id:a.class_id,
      shape:(a.shape as AnnShape)||'polygon',
      shape_params:a.shape_params||null,
      // 加载时也 clamp，避免历史 VOC 导入的越界数据被原样回写造成 422；
      // 再从原图坐标系换算到当前显示坐标系（viewRot=0 时为恒等）
      polygon:sanitizePolygon((a.polygon as any[]).map(p => Array.isArray(p) ? {x:p[0],y:p[1]} : {x:p.x,y:p.y})).map(origToDisp)
    })).filter(a=>a.polygon.length>=3)
    await nextTick()
    if(reqImageId!==currentImageId.value)return  // nextTick 后还得再确认一次
    renderAnnotations()
  }catch{
    if(reqImageId===currentImageId.value) annotations.value=[]
  }
  if(reqImageId!==currentImageId.value)return
  dirty.value=false;undoStack.value=[];redoStack.value=[];selectedAnnIdx.value=-1
}

function renderAnnotations(){
  if(!canvas||!currentImage.value)return
  canvas.getObjects().filter(o=>(o as any)._ann).forEach(o=>canvas!.remove(o))
  if(isSegTask.value&&!showAnnotationLayer.value){
    removeVertexHandles()
    canvas.requestRenderAll()
    return
  }
  const W=dispW(),H=dispH()
  // select 工具下让 polygon 可拖动整体平移（其他工具下保持 evented=true 用于点击选中，但禁拖）
  const isSelectTool=currentTool.value==='select'
  annotations.value.forEach((ann,idx)=>{
    const c=clsColor(ann.class_id)
    const pts=ann.polygon.map(p=>({x:p.x*W,y:p.y*H}))
    const poly=new fabric.Polygon(pts,{
      fill:c+'55',stroke:c,strokeWidth:0.75,
      selectable:isSelectTool,
      hasControls:false,hasBorders:false,
      lockRotation:true,lockScalingX:true,lockScalingY:true,
      hoverCursor:isSelectTool?'move':'default',
      objectCaching:false,
    })
    ;(poly as any)._ann=true
    ;(poly as any)._idx=idx
    // 整体拖动：mousedown 记录原点，moving 让所有 handle 跟随，modified 应用偏移到数据
    let dragOrigin:{left:number,top:number}|null=null
    let handleOrigin:{h:fabric.Circle|fabric.Line;left:number;top:number}[]=[]
    poly.on('mousedown',()=>{
      if(currentTool.value!=='select')return
      dragOrigin={left:poly.left||0,top:poly.top||0}
    })
    poly.on('moving',()=>{
      if(!dragOrigin||!currentImage.value)return
      const dx=(poly.left||0)-dragOrigin.left
      const dy=(poly.top||0)-dragOrigin.top
      // 让所有 handle（顶点/旋转点/连线）整体跟随。
      // 注意：必须在首个 moving 事件才记录初始位置（此时 selectAnn 触发的
      // handles 重建已完成），且不能按 editHandles 下标去取 polygon 顶点——
      // editHandles 前两个是虚线连线与旋转点，按下标取会整体错位。
      if(handleOrigin.length===0){
        handleOrigin=editHandles.map(h=>({h,left:h.left||0,top:h.top||0}))
      }
      handleOrigin.forEach(o=>{o.h.set({left:o.left+dx,top:o.top+dy});o.h.setCoords()})
      liveHint(`平移 Δx=${Math.round(dx)}px Δy=${Math.round(dy)}px`)
      canvas!.requestRenderAll()
    })
    poly.on('modified',()=>{
      dragHint.value=''
      if(!dragOrigin||!currentImage.value){handleOrigin=[];return}
      const dx=(poly.left||0)-dragOrigin.left
      const dy=(poly.top||0)-dragOrigin.top
      dragOrigin=null;handleOrigin=[]
      // 阈值过滤：< 0.5px 视为没动（避免单击也触发 dirty）
      if(Math.abs(dx)<0.5&&Math.abs(dy)<0.5)return
      pushUndo()
      const W2=dispW(),H2=dispH()
      annotations.value[idx].polygon=annotations.value[idx].polygon.map(p=>({
        x:Math.max(0,Math.min(1,p.x+dx/W2)),
        y:Math.max(0,Math.min(1,p.y+dy/H2)),
      }))
      dirty.value=true
      scheduleAutoSave()
      // 重建 polygon（让 fabric 的 left/top 复位）+ 重建 handles
      renderAnnotations()
      if(selectedAnnIdx.value===idx) showVertexHandles(idx)
      highlightSelected()
    })
    canvas!.add(poly)
  })
  canvas.requestRenderAll()
}

// ====== 多边形 ======
function onPolyClick(p:{x:number;y:number}){
  if(!isDrawing.value){isDrawing.value=true;drawingPoints=[{x:p.x,y:p.y}]}
  else drawingPoints.push({x:p.x,y:p.y})
  drawDotFixed(p.x,p.y)
}
async function finishPolygon(){
  if(!currentImage.value||drawingPoints.length<3){cancelDrawing();return}
  if(!(await ensureCanAddDefect())){cancelDrawing();return}
  const W=dispW(),H=dispH()
  addAnn(drawingPoints.map(p=>({x:p.x/W,y:p.y/H})));cancelDrawing()
}

// ====== 涂抹 ======
async function finishBrush(append:boolean=false){
  isBrushing=false;cleanTmp()
  if(!currentImage.value||brushPts.length<2)return
  if(!(await ensureCanAddDefect())){brushPts=[];return}
  const W=dispW(),H=dispH()
  // Shift 追加：拼接上次笔迹 + 本次笔迹，重新 rasterContour 拟合（rasterContour
  // 会在两笔末尾与下一笔开头之间画连线，自动桥接两段）。仅当上次 brush
  // annotation 仍在且同类时生效。
  if(append&&lastBrushAnnIdx>=0&&lastBrushAnnIdx<annotations.value.length
     &&annotations.value[lastBrushAnnIdx].class_id===selectedClassId.value
     &&lastBrushPts.length>0){
    const combined=[...lastBrushPts,...brushPts]
    const poly=rasterContour(combined,brushSize.value,W,H)
    if(poly.length>=3){
      pushUndo()
      annotations.value[lastBrushAnnIdx]={class_id:selectedClassId.value,polygon:sanitizePolygon(poly)}
      lastBrushPts=combined
      dirty.value=true
      renderAnnotations()
      scheduleAutoSave()
      brushPts=[]
      return
    }
  }
  const poly=rasterContour(brushPts,brushSize.value,W,H)
  if(poly.length>=3){
    addAnn(poly)
    lastBrushAnnIdx=annotations.value.length-1
    lastBrushPts=[...brushPts]
  }
  brushPts=[]
}

// ====== 折线 ======
function onPolylineClick(p:{x:number;y:number}){
  if(!isDrawing.value){isDrawing.value=true;drawingPoints=[{x:p.x,y:p.y}]}
  else drawingPoints.push({x:p.x,y:p.y})
  drawDotFixed(p.x,p.y)
}
async function finishPolyline(){
  if(!currentImage.value||drawingPoints.length<2){cancelDrawing();return}
  if(!(await ensureCanAddDefect())){cancelDrawing();return}
  const W=dispW(),H=dispH()
  const poly=rasterContour(drawingPoints,brushSize.value,W,H)
  if(poly.length>=3)addAnn(poly);cancelDrawing()
}

// ====== 圆形 ======
function drawCirclePrev(p:{x:number;y:number}){
  if(!canvas||!circleStart)return;cleanTmp()
  const c=clsColor(selectedClassId.value)
  const r=Math.sqrt((p.x-circleStart.x)**2+(p.y-circleStart.y)**2)
  const ci=new fabric.Circle({left:circleStart.x-r,top:circleStart.y-r,radius:r,fill:c+'33',stroke:c,strokeWidth:1,selectable:false,evented:false})
  ;(ci as any)._tmp=true;canvas.add(ci);canvas.requestRenderAll()
}
async function finishCircle(p:{x:number;y:number}){
  if(!currentImage.value||!circleStart){circleStart=null;cleanTmp();return}
  const r=Math.sqrt((p.x-circleStart.x)**2+(p.y-circleStart.y)**2)
  if(r<2){circleStart=null;cleanTmp();return}
  if(!(await ensureCanAddDefect())){circleStart=null;cleanTmp();return}
  const W=dispW(),H=dispH()
  addAnn(circlePolyFrom(circleStart.x,circleStart.y,r,W,H),'circle')
  circleStart=null;cleanTmp()
}

// ---- 3 点共圆 ----
/** 3 点共圆时的第二点预览：用「已点两点 + 当前鼠标」的外接圆做虚线预览 */
function drawCirclePrev3(p:{x:number;y:number}){
  if(!canvas||circlePts.length!==2)return;cleanTmp()
  const cc=circumcircle(circlePts[0],circlePts[1],p)
  if(!cc||cc.r<1)return
  const c=clsColor(selectedClassId.value)
  const ci=new fabric.Circle({
    left:cc.cx-cc.r,top:cc.cy-cc.r,radius:cc.r,
    fill:'transparent',stroke:c,strokeWidth:1,strokeDashArray:[5,4],
    selectable:false,evented:false,
  })
  ;(ci as any)._tmp=true;canvas.add(ci);canvas.requestRenderAll()
}
async function onCircle3Click(p:{x:number;y:number}){
  drawDotFixed(p.x,p.y)
  circlePts.push({x:p.x,y:p.y})
  circleClicks.value=circlePts.length
  if(circlePts.length<3)return
  const[a,b,c]=circlePts
  circlePts=[]
  circleClicks.value=0
  const cc=circumcircle(a,b,c)
  if(!cc||cc.r<2){cleanTmp();ElMessage.warning('三点近似共线，无法确定一个圆，请重新点击');return}
  if(!(await ensureCanAddDefect())){cleanTmp();return}
  const W=dispW(),H=dispH()
  addAnn(circlePolyFrom(cc.cx,cc.cy,cc.r,W,H),'circle')
  cleanTmp()
}

// ====== 矩形 ======
function drawRectPrev(p:{x:number;y:number}){
  if(!canvas||!rectStart)return;cleanTmp()
  const c=clsColor(selectedClassId.value)
  const x=Math.min(rectStart.x,p.x),y=Math.min(rectStart.y,p.y)
  const w=Math.abs(p.x-rectStart.x),h=Math.abs(p.y-rectStart.y)
  const r=new fabric.Rect({left:x,top:y,width:w,height:h,fill:c+'33',stroke:c,strokeWidth:1,selectable:false,evented:false})
  ;(r as any)._tmp=true;canvas.add(r);canvas.requestRenderAll()
}
async function finishRect(p:{x:number;y:number}){
  if(!currentImage.value||!rectStart){rectStart=null;cleanTmp();return}
  const W=dispW(),H=dispH()
  const x1=Math.min(rectStart.x,p.x),y1=Math.min(rectStart.y,p.y)
  const x2=Math.max(rectStart.x,p.x),y2=Math.max(rectStart.y,p.y)
  if(x2-x1<3||y2-y1<3){rectStart=null;cleanTmp();return} // 太小忽略
  // 4 点多边形：左上 → 右上 → 右下 → 左下（shape=rect，编辑时按矩形拖角点而不是拖顶点）
  if(!(await ensureCanAddDefect())){rectStart=null;cleanTmp();return}
  addAnn(rectPolyFrom(x1,y1,x2,y2,W,H),'rect')
  rectStart=null;cleanTmp()
}

// ====== 像素级擦除 ======
function finishEraser(){
  isErasing=false;cleanTmp()
  if(eraserPts.length<2)return
  subtractEraserFromAnnotations(eraserPts,brushSize.value)
  eraserPts=[]
}

// ====== 框选擦除 ======
function drawBoxRect(p:{x:number;y:number}){
  if(!canvas||!boxEraserStart)return;cleanTmp()
  const x=Math.min(boxEraserStart.x,p.x),y=Math.min(boxEraserStart.y,p.y)
  const r=new fabric.Rect({left:x,top:y,width:Math.abs(p.x-boxEraserStart.x),height:Math.abs(p.y-boxEraserStart.y),fill:'rgba(255,0,0,0.15)',stroke:'#F44',strokeWidth:0.5,strokeDashArray:[4,4],selectable:false,evented:false})
  ;(r as any)._tmp=true;canvas.add(r);canvas.requestRenderAll()
}
function finishBoxEraser(){
  if(!canvas||!boxEraserStart||!currentImage.value){boxEraserStart=null;cleanTmp();return}
  const W=dispW(),H=dispH()
  const t=canvas.getObjects().find(o=>(o as any)._tmp&&o.type==='rect') as fabric.Rect
  if(!t){boxEraserStart=null;return}
  const l=(t.left||0)/W,tp=(t.top||0)/H,r=l+(t.width||0)/W,b=tp+(t.height||0)/H
  const rm:number[]=[];annotations.value.forEach((a,i)=>{if(a.polygon.some(p=>p.x>=l&&p.x<=r&&p.y>=tp&&p.y<=b))rm.push(i)})
  if(rm.length>0){pushUndo();for(let i=rm.length-1;i>=0;i--)annotations.value.splice(rm[i],1);dirty.value=true;selectedAnnIdx.value=-1;removeVertexHandles();scheduleAutoSave()}
  boxEraserStart=null;cleanTmp();renderAnnotations()
}

// ====== 标记OK / 取消OK ======
function setLocalImageStatus(status:ImageInfo['status'], annotationCount=annotations.value.length){
  if(currentImage.value){
    currentImage.value.status=status
    currentImage.value.annotation_count=annotationCount
  }
  const il=imageList.value.find(i=>i.id===currentImageId.value)
  if(il){
    il.status=status
    il.annotation_count=annotationCount
  }
}

async function ensureCanAddDefect(){
  if(currentImage.value?.status!=='reviewed')return true
  try{
    await ElMessageBox.confirm(
      '当前图片已标记为 OK，添加缺陷会取消 OK 状态，并按缺陷样本参与训练。是否继续？',
      '取消 OK 标记',
      {confirmButtonText:'继续标注缺陷',cancelButtonText:'取消',type:'warning'},
    )
    setLocalImageStatus(annotations.value.length>0?'labeled':'unlabeled')
    return true
  }catch{
    return false
  }
}

async function markAsOk(){
  if(!currentImage.value)return
  if(annotations.value.length>0){
    try{
      await ElMessageBox.confirm(
        `当前已有 ${annotations.value.length} 个缺陷标注。标记 OK 会清空这些标注，并按 OK 负样本参与训练。是否继续？`,
        '标记 OK',
        {confirmButtonText:'清空并标记 OK',cancelButtonText:'取消',type:'warning'},
      )
    }catch{return}
  }
  // 清空标注并保存到数据库（OK=无缺陷的负样本）
  pushUndo(); annotations.value=[]; dirty.value=false
  await annotationApi.save(currentImageId.value,[])
  await imageApi.updateStatus(currentImageId.value,'reviewed')
  setLocalImageStatus('reviewed',0)
  renderAnnotations(); ElMessage.success('已标记为OK（负样本）')
}
async function unmarkOk(){
  if(!currentImage.value)return
  const newStatus=annotations.value.length>0?'labeled':'unlabeled'
  await imageApi.updateStatus(currentImageId.value,newStatus)
  setLocalImageStatus(newStatus)
  ElMessage.info('已取消OK标记')
}
async function clearAll(){
  try{await ElMessageBox.confirm('确定清空所有标注？','清空',{confirmButtonText:'清空',cancelButtonText:'取消',type:'warning'})}catch{return}
  pushUndo();annotations.value=[];dirty.value=true;selectedAnnIdx.value=-1;removeVertexHandles();renderAnnotations();scheduleAutoSave()
}

// ====== 通用 ======
/**
 * 将顶点 clamp 到 [0,1]，并去除连续重复点。
 * 确保后端校验 (PointSchema ge=0 le=1) 不会因越界报 422。
 */
function sanitizePolygon(polygon:Point[]):Point[]{
  const out:Point[]=[]
  for(const p of polygon){
    const x=Math.max(0,Math.min(1,Number(p.x)||0))
    const y=Math.max(0,Math.min(1,Number(p.y)||0))
    // 去重相邻完全相同点（避免 0 长度边）
    const last=out[out.length-1]
    if(!last||Math.abs(last.x-x)>1e-6||Math.abs(last.y-y)>1e-6) out.push({x,y})
  }
  return out
}

function addAnn(polygon:Point[],shape:AnnShape='polygon'){
  const cleaned=sanitizePolygon(polygon)
  if(cleaned.length<3)return // 至少 3 点
  pushUndo()
  // rect / circle 只记 shape 类型，几何参数一律从 polygon 反推（polygon 是唯一真相），
  // 这样整体平移、旋转、剪贴板、撤销都不用维护第二份数据
  annotations.value.push({class_id:selectedClassId.value,polygon:cleaned,shape,shape_params:shapeParamsFromPolygon(shape,cleaned,dispW(),dispH())})
  setLocalImageStatus('labeled',annotations.value.length)
  dirty.value=true
  renderAnnotations()
  scheduleAutoSave()
}

// ====== 参数化形状（矩形 / 圆形）======
// 设计：`polygon` 始终是几何唯一真相（训练/推理只认它），`shape` 只声明“这是矩形/圆”，
// 编辑时按形状给手柄（矩形拖 4 角、圆拖圆心/半径），改完重新生成 polygon。
// 这样不会把圆/矩形退化成“一堆可随意拖的顶点”。
const CIRCLE_SAMPLE=48
const MANUAL_HANDLE_R=4

/** 圆（显示坐标：中心归一化 + 像素半径）→ 归一化轮廓点 */
function circlePolyFrom(cx:number,cy:number,rPx:number,W:number,H:number):Point[]{
  const pts:Point[]=[]
  for(let i=0;i<CIRCLE_SAMPLE;i++){
    const a=2*Math.PI*i/CIRCLE_SAMPLE
    pts.push({x:(cx+rPx*Math.cos(a))/W,y:(cy+rPx*Math.sin(a))/H})
  }
  return pts
}

/** 矩形四角（左上 → 右上 → 右下 → 左下），与绘制顺序一致 */
function rectPolyFrom(x1:number,y1:number,x2:number,y2:number,W:number,H:number):Point[]{
  return [
    {x:x1/W,y:y1/H},{x:x2/W,y:y1/H},{x:x2/W,y:y2/H},{x:x1/W,y:y2/H},
  ]
}

/** 从归一化轮廓点反推形状参数（坐标单位与传入的 W/H 一致） */
function shapeParamsFromPolygon(shape:AnnShape|undefined,polygon:Point[],W:number,H:number):ShapeParams|undefined{
  if(shape!=='rect'&&shape!=='circle')return undefined
  if(!polygon.length)return undefined
  const xs=polygon.map(p=>p.x),ys=polygon.map(p=>p.y)
  const x1=Math.min(...xs),x2=Math.max(...xs),y1=Math.min(...ys),y2=Math.max(...ys)
  if(shape==='rect')return{x1,y1,x2,y2}
  return{
    cx:(x1+x2)/2,cy:(y1+y2)/2,
    r:((x2-x1)*W+(y2-y1)*H)/4,   // 圆在像素坐标下是正圆：宽高两个方向直径取均值 /2
  }
}

/** 当前显示坐标系下的形状参数 */
function dispShapeParams(ann:AnnotationData):ShapeParams|undefined{
  return shapeParamsFromPolygon(ann.shape,ann.polygon,dispW(),dispH())
}

/** 三点外接圆（像素坐标）；返回 null 表示三点近似共线 */
function circumcircle(a:Point,b:Point,c:Point):{cx:number;cy:number;r:number}|null{
  const d=2*(a.x*(b.y-c.y)+b.x*(c.y-a.y)+c.x*(a.y-b.y))
  if(Math.abs(d)<1e-6)return null
  const a2=a.x*a.x+a.y*a.y,b2=b.x*b.x+b.y*b.y,c2=c.x*c.x+c.y*c.y
  const cx=(a2*(b.y-c.y)+b2*(c.y-a.y)+c2*(a.y-b.y))/d
  const cy=(a2*(c.x-b.x)+b2*(a.x-c.x)+c2*(b.x-a.x))/d
  return{cx,cy,r:Math.hypot(a.x-cx,a.y-cy)}
}

function drawDotFixed(x:number,y:number){
  if(!canvas)return;const c=clsColor(selectedClassId.value)
  const d=new fabric.Circle({left:x-2,top:y-2,radius:2,fill:'#FFF',stroke:c,strokeWidth:1,selectable:false,evented:false})
  ;(d as any)._tmp=true;canvas.add(d);canvas.requestRenderAll()
}
function drawDotAt(p:{x:number;y:number},color:string){
  if(!canvas)return
  const d=new fabric.Circle({left:p.x-brushSize.value/2,top:p.y-brushSize.value/2,radius:brushSize.value/2,fill:color,selectable:false,evented:false})
  ;(d as any)._tmp=true;canvas.add(d)
}
function drawTmpLine(p:{x:number;y:number}){
  if(!canvas||drawingPoints.length===0)return
  canvas.getObjects().filter(o=>(o as any)._tl).forEach(o=>canvas!.remove(o))
  const last=drawingPoints[drawingPoints.length-1],c=clsColor(selectedClassId.value)
  const ln=new fabric.Line([last.x,last.y,p.x,p.y],{stroke:c,strokeWidth:0.75,strokeDashArray:[5,3],selectable:false,evented:false})
  ;(ln as any)._tmp=true;(ln as any)._tl=true;canvas.add(ln);canvas.requestRenderAll()
}
function cancelDrawing(){finishManualDrag();isDrawing.value=false;drawingPoints=[];isBrushing=false;brushPts=[];isErasing=false;eraserPts=[];circleStart=null;circlePts=[];circleClicks.value=0;rectStart=null;boxEraserStart=null;lastBrushAnnIdx=-1;lastBrushPts=[];cleanTmp()}
function cleanTmp(){if(!canvas)return;canvas.getObjects().filter(o=>(o as any)._tmp).forEach(o=>canvas!.remove(o));canvas.requestRenderAll()}
function selectAnn(idx:number){
  selectedAnnIdx.value=idx
  highlightSelected()
  // 选中后显示顶点 handles（仅在 select 工具下，避免与其他工具的鼠标事件冲突）
  if(currentTool.value==='select') showVertexHandles(idx)
  else removeVertexHandles()
}

function highlightSelected(){
  if(!canvas)return
  const idx=selectedAnnIdx.value
  canvas.getObjects().forEach(o=>{
    if((o as any)._ann){
      const i=(o as any)._idx
      o.set({strokeWidth:i===idx?1.5:0.75,opacity:idx<0?1:(i===idx?1:0.7)})
    }
  })
  canvas.requestRenderAll()
}

// ====== 顶点编辑 + 整体旋转（仅 select 工具）======
/**
 * 矩形 / 圆形的参数化手柄：矩形拖 4 个角（对角固定，轴对齐），
 * 圆拖圆心（平移，半径不变）+ 拖半径手柄（改半径）。
 * 不给顶点手柄，避免把形状退化成“一堆可随意拖的顶点”。
 */
function showShapeHandles(idx:number){
  removeVertexHandles()
  if(!canvas||!currentImage.value)return
  if(isSegTask.value&&!showAnnotationLayer.value)return
  const ann=annotations.value[idx]
  if(!ann)return
  const W=dispW(),H=dispH()
  const params=dispShapeParams(ann)
  if(!params)return
  const R=MANUAL_HANDLE_R
  const mkHandle=(x:number,y:number,color:string,cursor:string)=>{
    const h=new fabric.Circle({
      left:x-R,top:y-R,radius:R,
      fill:color,stroke:'#409EFF',strokeWidth:1.5,
      hasControls:false,hasBorders:false,
      selectable:true,evented:true,hoverCursor:cursor,
      originX:'left',originY:'top',
      lockRotation:true,lockScalingX:true,lockScalingY:true,
    })
    ;(h as any)._handle=true
    ;(h as any)._annIdx=idx
    return h
  }

  if(ann.shape==='rect'){
    const x1=params.x1!*W,y1=params.y1!*H,x2=params.x2!*W,y2=params.y2!*H
    // [手柄 x,y, 对角点 x,y]
    const corners:Array<[number,number,number,number]>=[
      [x1,y1,x2,y2],[x2,y1,x1,y2],[x2,y2,x1,y1],[x1,y2,x2,y1],
    ]
    corners.forEach(([hx,hy,ox,oy])=>{
      const handle=mkHandle(hx,hy,'#fff','nwse-resize')
      let firstMove=true
      handle.on('mousedown',()=>{firstMove=true})
      handle.on('moving',()=>{
        if(!currentImage.value)return
        const mx=(handle.left||0)+R,my=(handle.top||0)+R
        const nx1=Math.max(0,Math.min(W,Math.min(ox,mx)))
        const nx2=Math.max(0,Math.min(W,Math.max(ox,mx)))
        const ny1=Math.max(0,Math.min(H,Math.min(oy,my)))
        const ny2=Math.max(0,Math.min(H,Math.max(oy,my)))
        if(nx2-nx1<3||ny2-ny1<3)return
        // 第一次真正改变尺寸才入栈，避免单击也 push undo
        if(firstMove){pushUndo();firstMove=false}
        ann.polygon=rectPolyFrom(nx1,ny1,nx2,ny2,W,H)
        syncAnnShape(idx)
        liveHint(`矩形 ${Math.round(nx2-nx1)}×${Math.round(ny2-ny1)} px`)
        canvas!.requestRenderAll()
      })
      handle.on('modified',()=>{
        dragHint.value=''
        if(firstMove)return
        dirty.value=true;scheduleAutoSave()
        renderAnnotations();showShapeHandles(idx);highlightSelected()
      })
      canvas!.add(handle);editHandles.push(handle)
    })
    canvas.requestRenderAll()
    return
  }

  // ---- 圆 ----
  const cx=params.cx!*W,cy=params.cy!*H,r=params.r!
  const cHandle=mkHandle(cx,cy,'#FFD666','move')
  let origPoly:Point[]|null=null
  let startPt={x:cx,y:cy}
  let cMoved=false
  cHandle.on('mousedown',()=>{
    origPoly=ann.polygon.map(p=>({...p}))
    startPt={x:(cHandle.left||0)+R,y:(cHandle.top||0)+R}
    cMoved=false
  })
  cHandle.on('moving',()=>{
    if(!origPoly)return
    const mx=(cHandle.left||0)+R,my=(cHandle.top||0)+R
    if(!cMoved){pushUndo();cMoved=true}
    const dx=mx-startPt.x,dy=my-startPt.y
    ann.polygon=origPoly.map(p=>({
      x:Math.max(0,Math.min(1,p.x+dx/W)),
      y:Math.max(0,Math.min(1,p.y+dy/H)),
    }))
    syncAnnShape(idx)
    liveHint(`圆心 (${Math.round(mx)}, ${Math.round(my)}) · 半径不变`)
    canvas!.requestRenderAll()
  })
  cHandle.on('modified',()=>{
    dragHint.value=''
    origPoly=null
    if(!cMoved)return
    dirty.value=true;scheduleAutoSave()
    renderAnnotations();showShapeHandles(idx);highlightSelected()
  })
  canvas.add(cHandle);editHandles.push(cHandle)

  const rHandle=mkHandle(cx+r,cy,'#409EFF','ew-resize')
  let rFirst=true
  rHandle.on('mousedown',()=>{rFirst=true})
  rHandle.on('moving',()=>{
    const mx=(rHandle.left||0)+R,my=(rHandle.top||0)+R
    const nr=Math.hypot(mx-cx,my-cy)
    if(nr<2)return
    if(rFirst){pushUndo();rFirst=false}
    ann.polygon=circlePolyFrom(cx,cy,nr,W,H)
    syncAnnShape(idx)
    liveHint(`半径 ${Math.round(nr)} px`)
    canvas!.requestRenderAll()
  })
  rHandle.on('modified',()=>{
    dragHint.value=''
    if(rFirst)return
    dirty.value=true;scheduleAutoSave()
    renderAnnotations();showShapeHandles(idx);highlightSelected()
  })
  canvas.add(rHandle);editHandles.push(rHandle)
  canvas.requestRenderAll()
}

function showVertexHandles(idx:number){
  removeVertexHandles()
  if(!canvas||!currentImage.value)return
  if(isSegTask.value&&!showAnnotationLayer.value)return
  const ann=annotations.value[idx]
  if(!ann)return
  // 矩形 / 圆形走参数化手柄，不按多边形顶点编辑
  if(ann.shape==='rect'||ann.shape==='circle'){showShapeHandles(idx);return}
  const W=dispW(),H=dispH()

  // ---- 旋转手柄（绿色圆点，在 polygon 上方）----
  const xsAll=ann.polygon.map(p=>p.x*W)
  const ysAll=ann.polygon.map(p=>p.y*H)
  const rotCx=xsAll.reduce((a,b)=>a+b,0)/xsAll.length
  const rotCy=ysAll.reduce((a,b)=>a+b,0)/ysAll.length
  const rotMinY=Math.min(...ysAll)
  const rotHandleY=Math.max(10,rotMinY-26)   // 上方 26 px，至少留 10px
  const rotR=7
  // 中心 → 手柄的虚线连接
  const stem=new fabric.Line([rotCx,rotCy,rotCx,rotHandleY],{
    stroke:'#52C41A',strokeWidth:1,strokeDashArray:[3,3],
    selectable:false,evented:false,
  })
  ;(stem as any)._handle=true
  canvas.add(stem)
  editHandles.push(stem)

  const rotHandle=new fabric.Circle({
    left:rotCx-rotR,top:rotHandleY-rotR,radius:rotR,
    fill:'#52C41A',stroke:'#fff',strokeWidth:1.5,
    hasControls:false,hasBorders:false,
    selectable:true,evented:true,
    hoverCursor:'crosshair',
    originX:'left',originY:'top',
    lockRotation:true,lockScalingX:true,lockScalingY:true,
  })
  ;(rotHandle as any)._handle=true
  ;(rotHandle as any)._rotHandle=true
  ;(rotHandle as any)._annIdx=idx
  // 缓存：mousedown 时记录原始 polygon + 起始角度，moving 时基于增量旋转避免累积误差
  let origPoly:Point[]|null=null
  let origAngle=0
  let rotMoved=false
  rotHandle.on('mousedown',()=>{
    origPoly=ann.polygon.map(p=>({x:p.x,y:p.y}))
    const sx=(rotHandle.left||0)+rotR-rotCx
    const sy=(rotHandle.top||0)+rotR-rotCy
    origAngle=Math.atan2(sy,sx)
    rotMoved=false
  })
  rotHandle.on('moving',(opt:any)=>{
    if(!origPoly||!currentImage.value)return
    rotMoved=true
    const W2=dispW(),H2=dispH()
    const hx=(rotHandle.left||0)+rotR
    const hy=(rotHandle.top||0)+rotR
    let angleNow=Math.atan2(hy-rotCy,hx-rotCx)
    // 按住 Shift 吸附到 15° 整数倍（相对本次拖动的起始角），便于精确摆正
    if(opt?.e?.shiftKey){
      const step=Math.PI/12
      angleNow=origAngle+Math.round((angleNow-origAngle)/step)*step
    }
    const delta=angleNow-origAngle
    const cosD=Math.cos(delta),sinD=Math.sin(delta)
    for(let i=0;i<origPoly.length;i++){
      const px=origPoly[i].x*W2-rotCx
      const py=origPoly[i].y*H2-rotCy
      const nx=rotCx+px*cosD-py*sinD
      const ny=rotCy+px*sinD+py*cosD
      annotations.value[idx].polygon[i]={
        x:Math.max(0,Math.min(1,nx/W2)),
        y:Math.max(0,Math.min(1,ny/H2)),
      }
    }
    // 实时更新 polygon 显示
    syncAnnShape(idx)
    // 手柄与虚线连线跟随（吸附后按吸附方向重算，视觉与数据一致）
    const hx2=rotCx+Math.cos(angleNow)*(rotCy-rotHandleY)
    const hy2=rotCy+Math.sin(angleNow)*(rotCy-rotHandleY)
    rotHandle.set({left:hx2-rotR,top:hy2-rotR})
    rotHandle.setCoords()
    stem.set({x1:rotCx,y1:rotCy,x2:hx2,y2:hy2})
    ;(stem as any).setCoords?.()
    liveHint(`旋转 ${(((delta*180/Math.PI)%360+360)%360).toFixed(1)}°`)
    canvas!.requestRenderAll()
  })
  rotHandle.on('modified',()=>{
    dragHint.value=''
    if(!rotMoved){origPoly=null;return}   // 没真的拖动，不入栈不触发保存
    pushUndo()  // 旋转完成入 undo（拖一次只入栈一次）
    origPoly=null
    dirty.value=true
    scheduleAutoSave()
    renderAnnotations()
    showVertexHandles(idx)   // 重建 handles 到新位置
    highlightSelected()
  })
  canvas.add(rotHandle)
  editHandles.push(rotHandle)

  // ---- 4 个角顶点 ----
  ann.polygon.forEach((p,i)=>{
    const handle=new fabric.Circle({
      left:p.x*W-3,top:p.y*H-3,radius:3,
      fill:'#fff',stroke:'#409EFF',strokeWidth:1,
      hasControls:false,hasBorders:false,
      selectable:true,evented:true,
      hoverCursor:'move',
      originX:'left',originY:'top',
      lockRotation:true,lockScalingX:true,lockScalingY:true,
    })
    ;(handle as any)._handle=true
    ;(handle as any)._annIdx=idx
    ;(handle as any)._ptIdx=i
    let firstMove=true
    handle.on('mousedown',()=>{firstMove=true})
    handle.on('moving',()=>{
      if(!currentImage.value)return
      // 第一次移动才入栈，避免单击也 push undo
      if(firstMove){pushUndo();firstMove=false}
      const W2=dispW(),H2=dispH()
      const cx=(handle.left||0)+3,cy=(handle.top||0)+3
      // clamp 到画布范围（防止越界后端 422）
      const nx=Math.max(0,Math.min(1,cx/W2))
      const ny=Math.max(0,Math.min(1,cy/H2))
      annotations.value[idx].polygon[i]={x:nx,y:ny}
      // 实时更新对应 polygon 形状（不全量重绘，性能更好）
      syncAnnShape(idx)
      // 提示顶点在原图上的像素坐标（换算回原图坐标系，视图旋转时也正确）
      const op=dispToOrig({x:nx,y:ny}),cimg=currentImage.value
      liveHint(`顶点 #${i+1} → 原图 (${Math.round(op.x*(cimg?.width||0))}, ${Math.round(op.y*(cimg?.height||0))})`)
      canvas!.requestRenderAll()
    })
    handle.on('modified',()=>{
      dragHint.value=''
      if(firstMove)return // 没真的拖动
      dirty.value=true;scheduleAutoSave()
      // 拖动结束后完整重建 polygon 形状 + handles，
      // 避免 fabric 内部 pathOffset/边界不一致导致后续选中显示错位
      renderAnnotations()
      showVertexHandles(idx)
      highlightSelected()
    })
    canvas!.add(handle)
    editHandles.push(handle)
  })
  canvas!.requestRenderAll()
}

function removeVertexHandles(){
  if(!canvas)return
  canvas.getObjects().filter(o=>(o as any)._handle).forEach(o=>canvas!.remove(o))
  editHandles=[]
}

// ====== 区域整体变换：精确旋转 / 微调平移 / Alt+拖拽平移 ======

/** 找到某个标注对应的 fabric 对象 */
function annObj(idx:number):fabric.Polygon|undefined{
  return canvas?.getObjects().find(o=>(o as any)._ann&&(o as any)._idx===idx) as fabric.Polygon|undefined
}

/**
 * 把 annotations[idx].polygon 的新坐标同步到画布上的 polygon 对象，
 * 避免全量 renderAnnotations（拖动过程中更高频、更省）。
 * fabric 的 Polygon 改 points 后必须重算包围盒，否则选中区域/手柄会显示错位。
 */
function syncAnnShape(idx:number){
  if(!canvas)return
  const polyObj=annObj(idx),ann=annotations.value[idx]
  if(!polyObj||!ann)return
  const W=dispW(),H=dispH()
  polyObj.set({points:ann.polygon.map(pt=>({x:pt.x*W,y:pt.y*H})) as any})
  ;(polyObj as any)._calcDimensions?.()
  polyObj.setCoords()
  polyObj.dirty=true
}

/** 画布底部的短提示（旋转角度 / 位移），到时自动消失 */
function flashHint(text:string,ms=1500){
  dragHint.value=text
  if(hintTimer)clearTimeout(hintTimer)
  hintTimer=setTimeout(()=>{dragHint.value=''},ms)
}
/** 拖动过程中的实时提示：带兜底超时，万一没收到 mouseup 也不会一直挂在画布上 */
function liveHint(text:string){flashHint(text,2500)}

/** 选中区域绕自身质心旋转指定角度（显示坐标系 → 写回归一化坐标） */
function rotateSelectedBy(deg:number){
  const idx=selectedAnnIdx.value
  const ann=idx>=0?annotations.value[idx]:null
  if(!ann){ElMessage.warning('请先用「选择」工具点击一个标注区域');return}
  if(ann.polygon.length<3)return
  // 矩形/圆是参数化形状：旋转会破坏“轴对齐矩形 / 正圆”的语义，不提供旋转
  if(ann.shape==='rect'){
    ElMessage.warning('矩形为轴对齐形状，不支持旋转；如需斜置请用多边形工具绘制')
    return
  }
  if(ann.shape==='circle'){
    ElMessage.info('圆形旋转后不变（圆心+半径定义），无需旋转')
    return
  }
  const W=dispW(),H=dispH()
  const n=ann.polygon.length
  const cx=ann.polygon.reduce((s,p)=>s+p.x*W,0)/n
  const cy=ann.polygon.reduce((s,p)=>s+p.y*H,0)/n
  pushUndo()
  const rad=deg*Math.PI/180,cosD=Math.cos(rad),sinD=Math.sin(rad)
  ann.polygon=ann.polygon.map(p=>{
    const px=p.x*W-cx,py=p.y*H-cy
    return{
      x:Math.max(0,Math.min(1,(cx+px*cosD-py*sinD)/W)),
      y:Math.max(0,Math.min(1,(cy+px*sinD+py*cosD)/H)),
    }
  })
  dirty.value=true
  renderAnnotations()
  if(currentTool.value==='select')showVertexHandles(idx)
  highlightSelected()
  flashHint(`已旋转 ${deg>0?'+':''}${deg}°`)
  scheduleAutoSave()
}

let lastNudgeAt=0
/** 方向键微调平移选中区域；连续按键合并为一次 undo */
function nudgeSelected(dx:number,dy:number){
  const idx=selectedAnnIdx.value
  const ann=idx>=0?annotations.value[idx]:null
  if(!ann)return
  const W=dispW(),H=dispH()
  const now=Date.now()
  if(now-lastNudgeAt>700)pushUndo()   // 连续微调只在第一次入栈
  lastNudgeAt=now
  ann.polygon=ann.polygon.map(p=>({
    x:Math.max(0,Math.min(1,p.x+dx/W)),
    y:Math.max(0,Math.min(1,p.y+dy/H)),
  }))
  dirty.value=true
  syncAnnShape(idx)
  if(currentTool.value==='select')showVertexHandles(idx)   // 手柄跟随
  canvas?.requestRenderAll()
  liveHint(`平移 Δx=${dx}px Δy=${dy}px`)
  scheduleAutoSave()
}

/**
 * Alt+拖拽平移（任意工具下生效）。
 * 用独立状态手动实现拖动，因为非「选择」工具下 polygon 是 selectable:false，
 * fabric 不会为该对象启动拖拽变换。
 */
function startManualDrag(idx:number,p:{x:number;y:number}){
  const ann=annotations.value[idx]
  if(!ann)return
  selectedAnnIdx.value=idx
  highlightSelected()
  manualDrag={idx,startX:p.x,startY:p.y,origPoly:ann.polygon.map(q=>({x:q.x,y:q.y})),moved:false}
}

function applyManualDrag(p:{x:number;y:number}){
  const md=manualDrag
  if(!md)return
  const ann=annotations.value[md.idx]
  if(!ann){manualDrag=null;return}
  const W=dispW(),H=dispH()
  const dx=p.x-md.startX,dy=p.y-md.startY
  if(!md.moved){
    if(Math.abs(dx)<1&&Math.abs(dy)<1)return   // 阈值：避免只是点击就入栈
    pushUndo();md.moved=true
  }
  ann.polygon=md.origPoly.map(q=>({
    x:Math.max(0,Math.min(1,q.x+dx/W)),
    y:Math.max(0,Math.min(1,q.y+dy/H)),
  }))
  syncAnnShape(md.idx)
  canvas?.requestRenderAll()
  dragHint.value=`平移 Δx=${Math.round(dx)}px Δy=${Math.round(dy)}px`
}

function finishManualDrag(){
  const md=manualDrag
  manualDrag=null
  dragHint.value=''
  if(!md||!md.moved)return
  dirty.value=true
  renderAnnotations()
  if(currentTool.value==='select')showVertexHandles(md.idx)
  highlightSelected()
  scheduleAutoSave()
}

/**
 * 改已画标注的类别：换颜色 + 重新计入类别统计，可撤销、自动保存。
 * 只改 class_id，不动 polygon。
 */
function changeAnnClass(idx:number,classId:number){
  const ann=annotations.value[idx]
  if(!ann||classId==null||ann.class_id===classId)return
  pushUndo()
  ann.class_id=classId
  dirty.value=true
  renderAnnotations()
  if(currentTool.value==='select'&&selectedAnnIdx.value===idx)showVertexHandles(idx)
  highlightSelected()
  scheduleAutoSave()
}

// ====== 复制粘贴 ======
function copyAnn(){
  if(selectedAnnIdx.value<0)return
  const ann=annotations.value[selectedAnnIdx.value]
  if(!ann)return
  // 剪贴板统一存原图坐标系，粘贴时再转当前显示坐标系（旋转前后复制粘贴不错位）
  // shape 一起复制，矩形/圆粘贴后仍然是矩形/圆（参数从轮廓反推，不用单独存）
  clipboard={class_id:ann.class_id,polygon:ann.polygon.map(dispToOrig),shape:ann.shape}
  ElMessage.info('已复制')
}
async function pasteAnn(){
  if(!clipboard||!currentImage.value)return
  if(!(await ensureCanAddDefect()))return
  const W=dispW(),H=dispH()
  // 像素偏移 20px，转归一化
  const dx=20/W,dy=20/H
  const newPoly=clipboard.polygon.map(origToDisp).map(p=>({
    x:Math.max(0,Math.min(1,p.x+dx)),
    y:Math.max(0,Math.min(1,p.y+dy)),
  }))
  pushUndo()
  annotations.value.push({class_id:clipboard.class_id,polygon:newPoly,shape:clipboard.shape||'polygon',shape_params:shapeParamsFromPolygon(clipboard.shape,newPoly,W,H)})
  setLocalImageStatus('labeled',annotations.value.length)
  dirty.value=true
  selectedAnnIdx.value=annotations.value.length-1
  renderAnnotations()
  if(currentTool.value==='select') showVertexHandles(selectedAnnIdx.value)
  scheduleAutoSave()
  ElMessage.success('已粘贴')
}
function delAnn(idx:number){pushUndo();annotations.value.splice(idx,1);dirty.value=true;selectedAnnIdx.value=-1;removeVertexHandles();renderAnnotations();scheduleAutoSave()}
function pushUndo(){undoStack.value.push(JSON.stringify(annotations.value));redoStack.value=[];if(undoStack.value.length>50)undoStack.value.shift()}
function undo(){if(!canUndo.value)return;redoStack.value.push(JSON.stringify(annotations.value));annotations.value=JSON.parse(undoStack.value.pop()!);dirty.value=true;selectedAnnIdx.value=-1;removeVertexHandles();renderAnnotations();scheduleAutoSave()}
function redo(){if(!canRedo.value)return;undoStack.value.push(JSON.stringify(annotations.value));annotations.value=JSON.parse(redoStack.value.pop()!);dirty.value=true;selectedAnnIdx.value=-1;removeVertexHandles();renderAnnotations();scheduleAutoSave()}

/**
 * 标注变化后调度自动保存（600ms 防抖）。
 * 频繁拖动顶点不会每次都触发请求，最后一次操作 600ms 后才发。
 */
function scheduleAutoSave(){
  if(!dirty.value) return
  saveState.value='pending'
  if(autoSaveTimer) clearTimeout(autoSaveTimer)
  autoSaveTimer=setTimeout(()=>{handleSave({silent:true})},600)
}

async function handleSave(opt:{silent?:boolean;immediate?:boolean}={}){
  if(autoSaveTimer){clearTimeout(autoSaveTimer);autoSaveTimer=null}
  if(saving.value) return // 上一次保存还在进行
  if(!dirty.value && !opt.immediate) return
  const imgId=currentImageId.value
  saving.value=true; saveState.value='saving'
  // 提交前：显示坐标系 → 原图坐标系（viewRot=0 时为恒等），
  // 再做兜底 sanitize：兼容历史 VOC 导入数据中的越界顶点
  // (后端 PointSchema 强制 [0,1]，越界值会 422)
  const safeAnns=annotations.value.map(a=>{
    const polygon=sanitizePolygon(a.polygon.map(dispToOrig))
    const shape:AnnShape=a.shape||'polygon'
    // 参数一律从保存后的轮廓点反推，且用**原图**宽高（圆半径是像素值，
    // 显示坐标系旋转 90° 后宽高互换，不能拿 dispW/dispH 算）
    const img=currentImage.value
    const params=shapeParamsFromPolygon(shape,polygon,img?.width||1,img?.height||1)
    return {...a,polygon,shape,shape_params:params}
  }).filter(a=>a.polygon.length>=3)
  try{
    await annotationApi.save(imgId,safeAnns)
    // 仅当当前图未切换时才更新状态（防止用户切图后异步回写覆盖新图状态）
    if(imgId===currentImageId.value){
      dirty.value=false
      saveState.value='saved'
      // 2s 后清掉"已保存"提示
      setTimeout(()=>{if(saveState.value==='saved')saveState.value='idle'},2000)
    }
    const il=imageList.value.find(i=>i.id===imgId)
    if(il){
      const cur=currentImage.value
      il.status=safeAnns.length>0?'labeled':'unlabeled'
      if(cur?.id===imgId){
        cur.status=il.status
      }
      il.annotation_count=safeAnns.length
    }
    if(!opt.silent) ElMessage.success('保存成功')
  }catch(e:any){
    saveState.value='error'
    if(!opt.silent) ElMessage.error('保存失败：'+(e?.response?.data?.detail||e?.message||''))
  }finally{saving.value=false}
}

async function switchImage(id:number){
  // 切图前 flush 自动保存（避免丢失最近改动）
  if(dirty.value){
    try{await handleSave({silent:true,immediate:true})}
    catch{
      try{await ElMessageBox.confirm('保存失败，仍要切换吗？','提示',{confirmButtonText:'丢弃切换',cancelButtonText:'留下',type:'warning'})}
      catch{return}
    }
  }
  removeVertexHandles();selectedAnnIdx.value=-1
  lastBrushAnnIdx=-1;lastBrushPts=[]
  currentImageId.value=id;await loadImageAndAnnotations()
}
function prevImage(){if(hasPrev.value)switchImage(imageList.value[currentImageIdx.value-1].id)}
function nextImage(){if(hasNext.value)switchImage(imageList.value[currentImageIdx.value+1].id)}
function goBack(){router.push(`/project/${props.projectId}`)}
function zoomFit(){
  if(!canvas||!canvasAreaRef.value)return
  const bg=canvas.getObjects().find(o=>(o as any)._bg);if(!bg)return
  // 用显示坐标系尺寸（旋转 90/270 时宽高互换），不能直接用 fabric 对象的 width/height
  const cw=canvasAreaRef.value.clientWidth,ch=canvasAreaRef.value.clientHeight
  const iw=dispW(),ih=dispH()
  const z=Math.min(cw/iw,ch/ih)*0.95
  // 一次性落地最终变换，并且必须用 setViewportTransform：
  // fabric 的 findTarget 用视口坐标（lineCoords）做命中判定，
  // 直接改 canvas.viewportTransform 而不刷新对象坐标，会导致
  // 「适应窗口」之后所有标注区域都点不中、拖不动（平移旋转失效的真因）。
  canvas.setViewportTransform([z,0,0,z,(cw-iw*z)/2,(ch-ih*z)/2])
  canvas.requestRenderAll();zoomLevel.value=z
}
function isTypingTarget(e:KeyboardEvent){
  // 只有真正的文本输入才算「正在输入」：工具栏的单选/复选/滑块(radio/checkbox/range)
  // 点完会保留焦点，若一并拦截会导致方向键微调、[ ] 旋转全部失效
  const t=e.target as HTMLElement|null
  if(!t)return false
  if(t.isContentEditable)return true
  const tag=t.tagName
  if(tag==='TEXTAREA'||tag==='SELECT')return true
  if(tag!=='INPUT')return false
  const type=((t as HTMLInputElement).type||'text').toLowerCase()
  return ['text','search','url','tel','email','password','number'].includes(type)
}
function onKD(e:KeyboardEvent){
  if(e.code==='Space'){e.preventDefault();spaceDown.value=true}
  else if(e.key==='Escape')cancelDrawing()
  else if(e.key==='Enter'&&isDrawing.value){if(currentTool.value==='polygon')finishPolygon();else if(currentTool.value==='polyline')finishPolyline()}
  else if(e.ctrlKey&&e.key==='z'){e.preventDefault();undo()}
  else if(e.ctrlKey&&e.key==='y'){e.preventDefault();redo()}
  else if(e.ctrlKey&&e.key==='s'){e.preventDefault();handleSave({immediate:true})}
  else if(e.ctrlKey&&(e.key==='c'||e.key==='C')){e.preventDefault();copyAnn()}
  else if(e.ctrlKey&&(e.key==='v'||e.key==='V')){e.preventDefault();pasteAnn()}
  // 选中区域的精确微调 / 旋转（放在方向键切图之前判断，Shift+方向键才微调）
  else if(selectedAnnIdx.value>=0&&e.shiftKey&&(e.key==='ArrowLeft'||e.key==='ArrowRight'||e.key==='ArrowUp'||e.key==='ArrowDown')){
    if(isTypingTarget(e))return
    e.preventDefault()
    const step=e.ctrlKey?10:1
    nudgeSelected(e.key==='ArrowLeft'?-step:(e.key==='ArrowRight'?step:0),e.key==='ArrowUp'?-step:(e.key==='ArrowDown'?step:0))
  }
  else if(selectedAnnIdx.value>=0&&(e.key==='['||e.key===']')){
    if(isTypingTarget(e))return
    e.preventDefault()
    rotateSelectedBy((e.key===']'?1:-1)*(e.shiftKey?1:15))
  }
  else if(!e.ctrlKey&&(e.key==='a'||e.key==='ArrowLeft'))prevImage()
  else if(!e.ctrlKey&&(e.key==='d'||e.key==='ArrowRight'))nextImage()
  else if(!e.ctrlKey&&(e.key==='r'||e.key==='R'))rotateView(90)
}
function onKU(e:KeyboardEvent){if(e.code==='Space')spaceDown.value=false}
function getThumbUrl(id:number){return imageApi.getFileUrl(id,true)}
function clsColor(cid:number){return defectClasses.value.find(c=>c.id===cid)?.color||'#FF0000'}
function clsName(cid:number){return defectClasses.value.find(c=>c.id===cid)?.name||'未知'}
function cntCls(cid:number){return annotations.value.filter(a=>a.class_id===cid).length}
function stText(s:string){return({unlabeled:'未标注',labeling:'标注中',labeled:'已标注',reviewed:'OK'})[s]||s}
function imgSC(img:ImageInfo){return{'sc-u':img.status==='unlabeled','sc-l':img.status==='labeled','sc-r':img.status==='reviewed'}}
// resize 监听器在 onMounted / onBeforeUnmount 里配对管理（见上方），避免每次进入标注页都泄漏一个监听器
</script>

<style scoped>
.annotator{display:flex;flex-direction:column;height:100vh;background:#1a1a1a;color:#ddd;outline:none}
.toolbar{display:flex;align-items:center;justify-content:space-between;padding:6px 12px;background:#2d2d2d;border-bottom:1px solid #444;min-height:48px;gap:8px;flex-wrap:wrap}
.toolbar-left,.toolbar-center,.toolbar-right{display:flex;align-items:center;gap:6px}
.infer-model-select{width:220px}
.filename{font-weight:500;font-size:14px}
.brush-size-ctrl{display:flex;align-items:center;gap:6px}
.ctrl-label{font-size:12px;color:#aaa}.ctrl-value{font-size:12px;color:#aaa;min-width:36px}
.seg-view-controls{display:flex;align-items:center;gap:8px;margin-left:8px}
.channel-radio{white-space:nowrap}
.main-area{display:flex;flex:1;overflow:hidden}
.nav-panel{width:200px;background:#252525;border-right:1px solid #444;display:flex;flex-direction:column}
.nav-header{padding:10px 12px;font-size:13px;font-weight:600;border-bottom:1px solid #333;color:#aaa}
.nav-list{flex:1;overflow-y:auto}
.nav-item{display:flex;align-items:center;gap:8px;padding:6px 8px;cursor:pointer;border-bottom:1px solid #333;border-left:3px solid transparent;transition:all .15s}
.nav-item:hover{background:#333}.nav-item.active{background:#1a3a5c;border-left-color:#409EFF}
.nav-item.sc-u{border-left-color:#555}.nav-item.sc-l{border-left-color:#E6A23C}.nav-item.sc-r{border-left-color:#67C23A}
.nav-thumb{width:48px;height:48px;object-fit:cover;border-radius:4px;background:#333}
.nav-info{flex:1;min-width:0}.nav-filename{font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#ccc}
.nav-status-row{display:flex;align-items:center;gap:4px;margin-top:2px}
.status-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.dot-unlabeled{background:#555}.dot-labeling{background:#E6A23C}.dot-labeled{background:#E6A23C;box-shadow:0 0 4px #E6A23C}.dot-reviewed{background:#67C23A;box-shadow:0 0 4px #67C23A}
.nav-status-text{font-size:10px;color:#999}
.nav-ann-count{font-size:10px;background:#555;color:#eee;padding:0 5px;border-radius:8px;margin-left:auto}
.nav-footer{display:flex;gap:4px;padding:8px;border-top:1px solid #333}
.canvas-area{flex:1;position:relative;overflow:hidden}
.draw-hint{position:absolute;bottom:12px;left:50%;transform:translateX(-50%);background:rgba(0,0,0,.7);color:#eee;padding:6px 16px;border-radius:20px;font-size:12px;pointer-events:none}
.side-panel{width:240px;background:#252525;border-left:1px solid #444;display:flex;flex-direction:column}
.panel-section{border-bottom:1px solid #333}.panel-title{padding:10px 12px;font-size:13px;font-weight:600;color:#aaa}
.class-list{padding:0 8px 10px}
.class-item{display:flex;align-items:center;gap:8px;padding:8px 10px;border-radius:6px;cursor:pointer;transition:background .15s}
.class-item:hover{background:#333}.class-item.active{background:#1a3a5c}
.class-dot{width:14px;height:14px;border-radius:3px;flex-shrink:0}.class-name{flex:1;font-size:13px}.class-count{font-size:12px;color:#888}
.empty-hint{padding:20px;text-align:center;color:#666;font-size:13px}
.ann-item{display:flex;align-items:center;gap:6px;padding:6px 12px;cursor:pointer;transition:background .15s}
.ann-item:hover{background:#333}.ann-item.selected{background:#1a3a5c}
.ann-label{flex:1;font-size:12px}.ann-points{font-size:11px;color:#888}
.ann-class-select{flex:1;min-width:0}
.ann-class-select :deep(.el-select__wrapper){min-height:24px;padding:0 8px}
.ann-class-select :deep(.el-select__selected-item){font-size:12px}
.statusbar{display:flex;align-items:center;gap:24px;padding:4px 16px;background:#2d2d2d;border-top:1px solid #444;font-size:12px;color:#888}
.status-shortcuts{margin-left:auto;color:#666}
.save-indicator{font-size:12px;min-width:80px;text-align:right;transition:color .2s}
.save-indicator.idle{color:transparent}
.save-indicator.pending{color:#E6A23C}
.save-indicator.saving{color:#909399}
.save-indicator.saved{color:#67C23A}
.save-indicator.error{color:#F56C6C;font-weight:600}
</style>
