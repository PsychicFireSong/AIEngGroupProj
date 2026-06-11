"use client";

import {
  Activity,
  AlertTriangle,
  Camera,
  ChevronDown,
  ChevronRight,
  CheckCircle2,
  CircleDot,
  Cpu,
  Gauge,
  History,
  ImageUp,
  Info as InfoIcon,
  Layers,
  Link2,
  LayoutDashboard,
  Moon,
  Palette,
  Pause,
  Play,
  RefreshCcw,
  Server,
  Settings2,
  ShieldAlert,
  Sun,
  Upload,
  Video,
  Zap,
  ZoomIn,
  ZoomOut,
  type LucideIcon,
} from "lucide-react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { SAMPLE_IMAGES } from "@/lib/sampleImages";
import type { ClassSeverityRow, ClassConfidenceRow, AreaCoverageRow } from "@/components/dashboard-charts";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type DragEvent } from "react";

const SeverityPieChart = dynamic(
  () => import("@/components/dashboard-charts").then((mod) => mod.SeverityPieChart),
  { ssr: false },
);
const DefectBarChart = dynamic(
  () => import("@/components/dashboard-charts").then((mod) => mod.DefectBarChart),
  { ssr: false },
);
const MetricBarChart = dynamic(
  () => import("@/components/dashboard-charts").then((mod) => mod.MetricBarChart),
  { ssr: false },
);
const ConditionTrendChart = dynamic(
  () => import("@/components/dashboard-charts").then((mod) => mod.ConditionTrendChart),
  { ssr: false },
);
const ClassSeverityTable = dynamic(
  () => import("@/components/dashboard-charts").then((mod) => mod.ClassSeverityTable),
  { ssr: false },
);
const ClassConfidencePanel = dynamic(
  () => import("@/components/dashboard-charts").then((mod) => mod.ClassConfidencePanel),
  { ssr: false },
);
const AreaCoverageChart = dynamic(
  () => import("@/components/dashboard-charts").then((mod) => mod.AreaCoverageChart),
  { ssr: false },
);

type Severity = "minor" | "moderate" | "critical" | "uncertain" | "unknown";

type Detection = {
  id: string;
  className: string;
  confidence: number;
  severity: Severity;
  severityConfidence: number;
  priority: "Immediate" | "Planned" | "Monitor";
  action: string;
  bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  areaRatio: number;
};

type BackendVideoFrame = {
  id?: string;
  frameTimeSec: number;
  detections: number;
  maxConfidence: number;
  classes: string[];
  confidence: number;
  evidenceImage?: string;
  annotatedImage?: string;
  detectionItems?: Detection[];
  conditionIndex?: number;
  riskScore?: number;
};

type BackendMediaSource = {
  kind?: string;
  mediaKind?: string;
  frameTimeSec?: number;
  selectedFrameTimeSec?: number;
  sampleCount?: number;
  scanFps?: number;
  coverageSeconds?: number;
  safetyCapped?: boolean;
  sampledFrames?: BackendVideoFrame[];
  detectedFrames?: BackendVideoFrame[];
  positiveFrameCount?: number;
  returnedDefectFrames?: number;
  requestedConfidence?: number;
  usedConfidence?: number;
  reviewMode?: boolean;
  youtube?: boolean;
  googleDrive?: boolean;
};

type InferenceResult = {
  detections: Detection[];
  latencyMs: number;
  stageMetrics: {
    stage1LatencyMs: number;
    stage2LatencyMs: number;
    cropsClassified: number;
    detectorModel: string;
    severityModel: string;
    sampledFrames?: number;
    scanFps?: number;
    usedConfidence?: number;
    videoScanLatencyMs?: number;
  };
  conditionIndex: number;
  riskScore: number;
  evidenceImage?: string;
  annotatedImage?: string;
  source?: BackendMediaSource;
};

type VideoFrameSummary = {
  id: string;
  frameTimeSec: number;
  detections: number;
  maxConfidence: number;
  classes: string[];
  confidence: number;
  evidenceImage?: string;
  annotatedImage?: string;
  detectionItems?: Detection[];
  conditionIndex?: number;
  riskScore?: number;
  selected?: boolean;
};

type InspectionRecord = Omit<InferenceResult, "source"> & {
  id: string;
  createdAt: string;
  source: "camera" | "image" | "video";
  mediaSource?: BackendMediaSource;
  asset: string;
  preview?: string;
  sourceUrl?: string;
  frames?: VideoFrameSummary[];
  scannedFrames?: number;
};

type VideoJobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

type VideoJob = {
  id: string;
  status: VideoJobStatus;
  progress: number;
  stage: string;
  message: string;
  mediaUrl?: string;
  queuePosition?: number;
  totalFrames?: number | null;
  processedFrames?: number;
  positiveFrames?: number;
  returnedDefectFrames?: number;
  scanFps?: number;
  currentFrameTimeSec?: number;
  usedConfidence?: number;
  createdAt?: string;
  startedAt?: string;
  finishedAt?: string;
  error?: string;
  result?: InferenceResult;
};

type ModelOption = {
  label: string;
  value: string;
  kind: "detector" | "severity";
};

type Settings = {
  mode: "Near real-time" | "Balanced" | "High accuracy" | "Multi-defect review";
  confidence: number;
  iou: number;
  frameIntervalMs: number;
  detectorPath: string;
  severityPath: string;
};

type ThemeMode = "dark" | "light";
type PaletteName = "Signal" | "Civic" | "Hazard";
type WorkspaceView = "overview" | "monitor";
type AppPage = "dashboard" | "monitoring" | "analytics" | "history" | "settings";
type MonitorPanel = "findings" | "frames" | "selected" | "charts";

const API_BASE = process.env.NEXT_PUBLIC_INFERENCE_API_URL ?? "http://localhost:8000";

const severityColors: Record<Severity, string> = {
  minor: "#2dd4bf",
  moderate: "#facc15",
  critical: "#fb7185",
  uncertain: "#a78bfa",
  unknown: "#94a3b8",
};

const priorityClasses = {
  Immediate: "border-rose-400/40 bg-rose-400/10 text-rose-100",
  Planned: "border-amber-300/40 bg-amber-300/10 text-amber-100",
  Monitor: "border-teal-300/40 bg-teal-300/10 text-teal-100",
};

const priorityMeta: Record<Detection["priority"], { label: string; description: string; icon: LucideIcon }> = {
  Immediate: {
    label: "Immediate",
    description: "Safety-sensitive finding. Escalate to urgent maintenance or restrict access when needed.",
    icon: ShieldAlert,
  },
  Planned: {
    label: "Planned",
    description: "Repair planning required. Add to maintenance schedule with supporting visual evidence.",
    icon: AlertTriangle,
  },
  Monitor: {
    label: "Monitor",
    description: "Low-risk finding. Keep evidence in history and compare during the next inspection cycle.",
    icon: CircleDot,
  },
};

const balancedSettings: Omit<Settings, "detectorPath" | "severityPath"> = {
  mode: "Balanced",
  confidence: 0.3,
  iou: 0.45,
  frameIntervalMs: 900,
};

const modeSettings: Record<Settings["mode"], Omit<Settings, "mode" | "detectorPath" | "severityPath">> = {
  "Near real-time": {
    confidence: 0.25,
    iou: 0.45,
    frameIntervalMs: 500,
  },
  Balanced: {
    confidence: 0.3,
    iou: 0.45,
    frameIntervalMs: 900,
  },
  "High accuracy": {
    confidence: 0.45,
    iou: 0.38,
    frameIntervalMs: 1500,
  },
  "Multi-defect review": {
    confidence: 0.1,
    iou: 0.55,
    frameIntervalMs: 1200,
  },
};

const maxLinkedVideoFps = 30;

const surfaceThemes: Record<ThemeMode, Record<string, string>> = {
  dark: {
    "--app-bg": "#161616",
    "--app-text": "#f4f4f4",
    "--heading": "#ffffff",
    "--muted": "#a8a8a8",
    "--panel": "#262626",
    "--panel-soft": "#2d2d2d",
    "--panel-strong": "#0f0f0f",
    "--rail": "#161616",
    "--line": "rgba(244, 244, 244, 0.16)",
    "--input-bg": "#393939",
    "--evidence-bg": "#020406",
    "--accent-ink": "#ffffff",
  },
  light: {
    "--app-bg": "#f4f4f4",
    "--app-text": "#262626",
    "--heading": "#161616",
    "--muted": "#6f6f6f",
    "--panel": "#ffffff",
    "--panel-soft": "#f4f4f4",
    "--panel-strong": "#e0e0e0",
    "--rail": "#ffffff",
    "--line": "rgba(22, 22, 22, 0.14)",
    "--input-bg": "#ffffff",
    "--evidence-bg": "#111827",
    "--accent-ink": "#ffffff",
  },
};

const paletteThemes: Record<PaletteName, Record<string, string>> = {
  Signal: {
    "--accent": "#0f62fe",
    "--accent-soft": "rgba(15, 98, 254, 0.18)",
    "--accent-border": "rgba(15, 98, 254, 0.48)",
    "--secondary": "#08bdba",
    "--secondary-soft": "rgba(8, 189, 186, 0.16)",
    "--secondary-border": "rgba(8, 189, 186, 0.44)",
  },
  Civic: {
    "--accent": "#3b82f6",
    "--accent-soft": "rgba(59, 130, 246, 0.14)",
    "--accent-border": "rgba(59, 130, 246, 0.36)",
    "--secondary": "#22c55e",
    "--secondary-soft": "rgba(34, 197, 94, 0.14)",
    "--secondary-border": "rgba(34, 197, 94, 0.34)",
  },
  Hazard: {
    "--accent": "#f97316",
    "--accent-soft": "rgba(249, 115, 22, 0.15)",
    "--accent-border": "rgba(249, 115, 22, 0.38)",
    "--secondary": "#06b6d4",
    "--secondary-soft": "rgba(6, 182, 212, 0.14)",
    "--secondary-border": "rgba(6, 182, 212, 0.34)",
  },
};

const fallbackModels: ModelOption[] = [
  { label: "Select detector from API", value: "", kind: "detector" },
  { label: "Select severity classifier from API", value: "", kind: "severity" },
];

const emptyStageMetrics: InferenceResult["stageMetrics"] = {
  stage1LatencyMs: 0,
  stage2LatencyMs: 0,
  cropsClassified: 0,
  detectorModel: "not selected",
  severityModel: "not selected",
};

const navItems: { label: string; icon: LucideIcon; page: AppPage; href: string }[] = [
  { label: "Dashboard", icon: LayoutDashboard, page: "dashboard", href: "/" },
  { label: "Monitor", icon: Camera, page: "monitoring", href: "/monitoring" },
  { label: "Analytics", icon: Activity, page: "analytics", href: "/analytics" },
  { label: "History", icon: History, page: "history", href: "/history" },
  { label: "Settings", icon: Settings2, page: "settings", href: "/settings" },
];

const pageMeta: Record<AppPage, { eyebrow: string; title: string; subtitle: string }> = {
  dashboard: {
    eyebrow: "AI Facilities Operations",
    title: "Defect Intelligence Dashboard",
    subtitle: "Command overview for model readiness, evidence intake, processing queues, and recent inspection work.",
  },
  monitoring: {
    eyebrow: "Live Inspection",
    title: "Real-Time Monitoring Console",
    subtitle: "Camera, uploaded media, and queued video analysis with detection telemetry.",
  },
  analytics: {
    eyebrow: "Inspection Analytics",
    title: "Condition Trends and Defect Mix",
    subtitle: "Aggregate every saved inspection into trends, class balance, severity mix, latency, and risk movement.",
  },
  history: {
    eyebrow: "Evidence Archive",
    title: "Inspection Records",
    subtitle: "Browse saved evidence. Click Open result to focus one inspection in the evidence review workspace.",
  },
  settings: {
    eyebrow: "Model Operations",
    title: "Inference Settings",
    subtitle: "Choose models and tune confidence, IoU, and frame sampling for deployment use.",
  },
};

const monitorPanels: { id: MonitorPanel; label: string; icon: LucideIcon }[] = [
  { id: "findings", label: "Find", icon: ImageUp },
  { id: "frames", label: "Frames", icon: Video },
  { id: "selected", label: "Action", icon: AlertTriangle },
  { id: "charts", label: "Charts", icon: RefreshCcw },
];

function pct(value: number) {
  return `${Math.round(value * 100)}%`;
}

function scoreTone(value: number) {
  if (value >= 82) return "text-teal-200";
  if (value >= 62) return "text-amber-200";
  return "text-rose-200";
}

function formatMs(value: number) {
  if (!value) return "n/a";
  return `${Math.round(value)} ms`;
}

function workspaceForPage(page: AppPage): WorkspaceView {
  return page === "monitoring" || page === "settings" ? "monitor" : "overview";
}

function modelDisplayName(value: string) {
  if (!value) return "not selected";
  return value.split(/[\\/]/).pop() || value;
}

function preferredModelValue(models: ModelOption[], kind: ModelOption["kind"]) {
  const candidates = models.filter((item) => item.kind === kind && item.value);
  const preferredTerms =
    kind === "detector"
      ? [
          "stage1_robust_later_best.pt",
          "stage1_anchor_robust_curriculum",
          "defect_detector_anchor_balanced_candidate.pt",
          "stage1_robust_epoch39_best.pt",
          "stage1_clean_near_done_best.pt",
          "defect_detector.pt",
          "stage1_defect_detector",
        ]
      : ["severity_cls.pt", "severity_cls_domain_adapted.pt", "stage2_revised_severity", "stage2_severity"];
  return (
    candidates.find((item) => preferredTerms.some((term) => item.value.toLowerCase().includes(term)))?.value ??
    candidates[0]?.value ??
    ""
  );
}

function normalizeResult<T extends InferenceResult>(result: T): T {
  return {
    ...result,
    stageMetrics: {
      ...emptyStageMetrics,
      ...(result.stageMetrics ?? {}),
    },
  };
}

function recordToInferenceResult(record: InspectionRecord): InferenceResult {
  return normalizeResult({
    detections: record.detections,
    latencyMs: record.latencyMs,
    stageMetrics: record.stageMetrics,
    conditionIndex: record.conditionIndex,
    riskScore: record.riskScore,
    evidenceImage: record.evidenceImage,
    annotatedImage: record.annotatedImage ?? record.preview,
    source: record.mediaSource,
  });
}

function normalizeRecord(record: InspectionRecord): InspectionRecord {
  const mediaSource = record.mediaSource ?? ((typeof record.source === "object" ? record.source : undefined) as BackendMediaSource | undefined);
  return {
    ...record,
    stageMetrics: {
      ...emptyStageMetrics,
      ...(record.stageMetrics ?? {}),
    },
    source: typeof record.source === "string" ? record.source : "image",
    mediaSource,
    frames: record.frames ?? [],
  };
}

function emptyInferenceResult(): InferenceResult {
  return {
    detections: [],
    latencyMs: 0,
    stageMetrics: emptyStageMetrics,
    conditionIndex: 100,
    riskScore: 0,
  };
}

function compactHistoryForStorage(records: InspectionRecord[]): InspectionRecord[] {
  return records.slice(0, 16).map((record, index) => {
    const keepCleanEvidence = index < 6 && record.evidenceImage && record.evidenceImage.length < 850_000;
    const keepEmbeddedEvidence = index < 6 && record.annotatedImage && record.annotatedImage.length < 850_000;
    const keepPreview = index < 6 && record.preview && record.preview.length < 850_000 && !record.preview.startsWith("blob:");
    return {
      ...record,
      evidenceImage: keepCleanEvidence ? record.evidenceImage : undefined,
      annotatedImage: keepEmbeddedEvidence ? record.annotatedImage : undefined,
      preview: keepPreview ? record.preview : undefined,
      frames: record.frames?.slice(0, 80).map((frame, frameIndex) => {
        const keepFrameCleanEvidence = index < 4 && frameIndex < 24 && frame.evidenceImage && frame.evidenceImage.length < 650_000;
        const keepFrameEvidence = index < 4 && frameIndex < 24 && frame.annotatedImage && frame.annotatedImage.length < 650_000;
        return {
          ...frame,
          evidenceImage: keepFrameCleanEvidence ? frame.evidenceImage : undefined,
          annotatedImage: keepFrameEvidence ? frame.annotatedImage : undefined,
          detectionItems: frame.detectionItems?.slice(0, 60),
        };
      }),
    };
  });
}

function dashboardThemeStyle(themeMode: ThemeMode, paletteName: PaletteName) {
  return {
    ...surfaceThemes[themeMode],
    ...paletteThemes[paletteName],
  } as CSSProperties & Record<`--${string}`, string>;
}

function defectFrameSummaries(result: InferenceResult): VideoFrameSummary[] {
  const selectedFrame = result.source?.selectedFrameTimeSec;
  const sourceFrames =
    result.source?.detectedFrames && result.source.detectedFrames.length > 0
      ? result.source.detectedFrames
      : result.source?.sampledFrames ?? [];
  return sourceFrames
    .filter((frame) => frame.detections > 0)
    .map((frame, index) => ({
      ...frame,
      id: frame.id ?? `${frame.frameTimeSec}-${frame.classes.join("-")}-${frame.maxConfidence}-${index}`,
      selected: selectedFrame !== undefined && Math.abs(frame.frameTimeSec - selectedFrame) < 0.2,
    }));
}

function singleFrameSummary(result: InferenceResult): VideoFrameSummary | null {
  if (!result.detections.length) return null;
  return {
    id: crypto.randomUUID(),
    frameTimeSec: result.source?.selectedFrameTimeSec ?? result.source?.frameTimeSec ?? 0,
    detections: result.detections.length,
    maxConfidence: Math.max(...result.detections.map((item) => item.confidence)),
    classes: Array.from(new Set(result.detections.map((item) => item.className))),
    confidence: result.source?.usedConfidence ?? 0,
    selected: true,
  };
}

function detectionsForRecord(record: InspectionRecord): Detection[] {
  const frameDetections =
    record.source === "video"
      ? (record.frames ?? []).flatMap((frame) => frame.detectionItems ?? [])
      : [];
  return frameDetections.length ? frameDetections : record.detections;
}

function severityChartData(detections: Detection[]) {
  const counts = detections.reduce<Record<Severity, number>>(
    (acc, detection) => {
      acc[detection.severity] += 1;
      return acc;
    },
    { minor: 0, moderate: 0, critical: 0, uncertain: 0, unknown: 0 },
  );
  return Object.entries(counts)
    .filter(([, value]) => value > 0)
    .map(([name, value]) => ({ name, value, color: severityColors[name as Severity] }));
}

function defectChartData(detections: Detection[]) {
  const counts = detections.reduce<Record<string, number>>((acc, detection) => {
    acc[detection.className] = (acc[detection.className] ?? 0) + 1;
    return acc;
  }, {});
  return Object.entries(counts).map(([name, value]) => ({ name, value }));
}

function actionWorkloadData(detections: Detection[]) {
  const counts = detections.reduce<Record<Detection["priority"], number>>(
    (acc, detection) => {
      acc[detection.priority] += 1;
      return acc;
    },
    { Immediate: 0, Planned: 0, Monitor: 0 },
  );
  return [
    { name: "Immediate", value: counts.Immediate, color: "#fb7185" },
    { name: "Planned", value: counts.Planned, color: "#f59e0b" },
    { name: "Monitor", value: counts.Monitor, color: "#2dd4bf" },
  ].filter((item) => item.value > 0);
}

function conditionBandName(value: number) {
  if (value >= 82) return "Healthy";
  if (value >= 62) return "Watch";
  return "High risk";
}

function conditionBandData(records: InspectionRecord[], fallback: InferenceResult) {
  const source = records.length
    ? records.map((record) => record.conditionIndex)
    : [fallback.conditionIndex];
  const counts = source.reduce<Record<string, number>>(
    (acc, condition) => {
      acc[conditionBandName(condition)] += 1;
      return acc;
    },
    { Healthy: 0, Watch: 0, "High risk": 0 },
  );
  return [
    { name: "Healthy", value: counts.Healthy, color: "#2dd4bf" },
    { name: "Watch", value: counts.Watch, color: "#facc15" },
    { name: "High risk", value: counts["High risk"], color: "#fb7185" },
  ].filter((item) => item.value > 0);
}

function recommendationSummary(detections: Detection[]) {
  const immediate = detections.filter((item) => item.priority === "Immediate").length;
  const planned = detections.filter((item) => item.priority === "Planned").length;
  const monitor = detections.filter((item) => item.priority === "Monitor").length;
  const manualReview = detections.filter((item) => item.severity === "uncertain" || item.severity === "unknown" || item.confidence < 0.35).length;
  return { immediate, planned, monitor, manualReview };
}

function classSeverityData(detections: Detection[]): ClassSeverityRow[] {
  const map = new Map<string, ClassSeverityRow>();
  for (const d of detections) {
    if (!map.has(d.className)) {
      map.set(d.className, { className: d.className, minor: 0, moderate: 0, critical: 0, uncertain: 0 });
    }
    const row = map.get(d.className)!;
    const sev = d.severity === "unknown" ? "uncertain" : d.severity;
    if (sev === "minor" || sev === "moderate" || sev === "critical" || sev === "uncertain") {
      row[sev] += 1;
    } else {
      row.uncertain += 1;
    }
  }
  return Array.from(map.values()).sort((a, b) => (b.critical + b.moderate) - (a.critical + a.moderate));
}

function classConfidenceData(detections: Detection[]): ClassConfidenceRow[] {
  const acc = new Map<string, { sumDet: number; sumSev: number; count: number; uncertain: number }>();
  for (const d of detections) {
    if (!acc.has(d.className)) acc.set(d.className, { sumDet: 0, sumSev: 0, count: 0, uncertain: 0 });
    const entry = acc.get(d.className)!;
    entry.sumDet += d.confidence ?? 0;
    entry.sumSev += d.severityConfidence ?? 0;
    entry.count += 1;
    if (d.severity === "uncertain" || d.severity === "unknown" || (d.confidence ?? 1) < 0.35) {
      entry.uncertain += 1;
    }
  }
  return Array.from(acc.entries())
    .map(([className, e]): ClassConfidenceRow => ({
      className,
      avgDetConf: e.count ? e.sumDet / e.count : 0,
      avgSevConf: e.count ? e.sumSev / e.count : 0,
      count: e.count,
      uncertainRate: e.count ? e.uncertain / e.count : 0,
    }))
    .sort((a, b) => b.count - a.count);
}

function areaCoverageData(detections: Detection[]): AreaCoverageRow[] {
  const acc = new Map<string, { sum: number; max: number; count: number }>();
  for (const d of detections) {
    const area = d.areaRatio ?? 0;
    if (!acc.has(d.className)) acc.set(d.className, { sum: 0, max: 0, count: 0 });
    const entry = acc.get(d.className)!;
    entry.sum += area;
    if (area > entry.max) entry.max = area;
    entry.count += 1;
  }
  return Array.from(acc.entries())
    .map(([className, e]): AreaCoverageRow => ({
      className,
      avgArea: e.count ? e.sum / e.count : 0,
      maxArea: e.max,
      count: e.count,
    }))
    .sort((a, b) => b.avgArea - a.avgArea);
}

function previewForRecord(record: InspectionRecord) {
  return (
    record.evidenceImage ??
    record.annotatedImage ??
    record.preview ??
    record.frames?.find((frame) => frame.evidenceImage || frame.annotatedImage)?.evidenceImage ??
    record.frames?.find((frame) => frame.evidenceImage || frame.annotatedImage)?.annotatedImage ??
    ""
  );
}

export function ControlRoom({ page }: { page: AppPage }) {
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const intervalRef = useRef<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const cameraStreamRef = useRef<MediaStream | null>(null);
  const videoSessionIdRef = useRef<string>("");
  const completedVideoJobIdsRef = useRef<Set<string>>(new Set());
  const uploadMediaRef = useRef<((file: File | undefined) => Promise<void>) | null>(null);
  const monitorRef = useRef<HTMLDivElement | null>(null);
  const analyticsRef = useRef<HTMLDivElement | null>(null);
  const historyRef = useRef<HTMLDivElement | null>(null);

  const [apiOnline, setApiOnline] = useState(false);
  const [apiMessage, setApiMessage] = useState("Checking inference API");
  const [models, setModels] = useState<ModelOption[]>(fallbackModels);
  const [settings, setSettings] = useState<Settings>({
    ...balancedSettings,
    detectorPath: "",
    severityPath: "",
  });
  const [settingsPanelOpen, setSettingsPanelOpen] = useState(page === "settings");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(page === "monitoring" || page === "settings");
  const [workspaceView, setWorkspaceView] = useState<WorkspaceView>(() => workspaceForPage(page));
  const [themeMode, setThemeMode] = useState<ThemeMode>("dark");
  const [paletteName, setPaletteName] = useState<PaletteName>("Signal");
  const [monitorPanel, setMonitorPanel] = useState<MonitorPanel>("findings");
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraMessage, setCameraMessage] = useState("Camera preview is off.");
  const [cameraError, setCameraError] = useState("");
  const [scanning, setScanning] = useState(false);
  const [busy, setBusy] = useState(false);
  const [sourceType, setSourceType] = useState<InspectionRecord["source"]>("camera");
  const [imagePreview, setImagePreview] = useState<string>("");
  const [videoObjectUrl, setVideoObjectUrl] = useState<string>("");
  const [showLinkInput, setShowLinkInput] = useState(false);
  const [mediaUrl, setMediaUrl] = useState("");
  const [linkFrameTime, setLinkFrameTime] = useState(0);
  const [scanLinkedVideo, setScanLinkedVideo] = useState(true);
  const [linkScanFps, setLinkScanFps] = useState(1);
  const [videoJob, setVideoJob] = useState<VideoJob | null>(null);
  const [evidenceZoom, setEvidenceZoom] = useState(1);
  const [latest, setLatest] = useState<InferenceResult>(() => emptyInferenceResult());
  const [selectedDetectionId, setSelectedDetectionId] = useState<string>("");
  const [history, setHistory] = useState<InspectionRecord[]>([]);
  const [historyReady, setHistoryReady] = useState(false);
  const [expandedHistoryId, setExpandedHistoryId] = useState<string>("");
  const [analysisStartedAt, setAnalysisStartedAt] = useState<number | null>(null);
  const [analysisElapsedSec, setAnalysisElapsedSec] = useState(0);
  const [dragActive, setDragActive] = useState(false);
  const [riskInfoOpen, setRiskInfoOpen] = useState(false);

  const detectorModels = models.filter((model) => model.kind === "detector");
  const severityModels = models.filter((model) => model.kind === "severity");
  const selectedDetection = latest.detections.find((item) => item.id === selectedDetectionId) ?? latest.detections[0];
  const latestDefectFrames = useMemo(() => defectFrameSummaries(latest), [latest]);
  const videoJobId = videoJob?.id ?? "";
  const activeVideoJob = videoJob?.status === "queued" || videoJob?.status === "running";
  const themeStyle = useMemo(() => dashboardThemeStyle(themeMode, paletteName), [themeMode, paletteName]);
  const currentPageMeta = pageMeta[page];
  const dashboardMode = page === "dashboard";
  const analyticsMode = page === "analytics";
  const historyMode = page === "history";
  const aggregateOverviewMode = dashboardMode || analyticsMode || historyMode;

  const analyticsDetections = useMemo(() => {
    if (!history.length) return latest.detections;
    return history.flatMap((record) => detectionsForRecord(record));
  }, [history, latest.detections]);
  const chartDetections = aggregateOverviewMode ? analyticsDetections : latest.detections;
  const severityData = useMemo(() => severityChartData(chartDetections), [chartDetections]);
  const defectData = useMemo(() => defectChartData(chartDetections), [chartDetections]);
  const actionData = useMemo(() => actionWorkloadData(chartDetections), [chartDetections]);
  const conditionData = useMemo(
    () => (aggregateOverviewMode && history.length === 0 ? [] : conditionBandData(history, latest)),
    [aggregateOverviewMode, history, latest],
  );
  const recommendationMetrics = useMemo(() => recommendationSummary(chartDetections), [chartDetections]);
  const classSevData = useMemo(() => classSeverityData(chartDetections), [chartDetections]);
  const classConfData = useMemo(() => classConfidenceData(chartDetections), [chartDetections]);
  const areaData = useMemo(() => areaCoverageData(chartDetections), [chartDetections]);

  const trendData = useMemo(() => {
    if (history.length === 0) {
      return aggregateOverviewMode ? [] : [{ label: "Now", condition: latest.conditionIndex, risk: latest.riskScore }];
    }
    return history.slice().reverse().map((item, index) => ({
      label: `S${index + 1}`,
      condition: item.conditionIndex,
      risk: item.riskScore,
    }));
  }, [aggregateOverviewMode, history, latest.conditionIndex, latest.riskScore]);

  const criticalCount = chartDetections.filter((item) => item.severity === "critical").length;
  const immediateCount = chartDetections.filter((item) => item.priority === "Immediate").length;
  const signalCondition =
    aggregateOverviewMode && history.length
      ? history.reduce((sum, record) => sum + record.conditionIndex, 0) / history.length
      : latest.conditionIndex;
  const signalLatency =
    aggregateOverviewMode && history.length
      ? history.reduce((sum, record) => sum + record.latencyMs, 0) / history.length
      : latest.latencyMs;

  useEffect(() => {
    let cancelled = false;
    queueMicrotask(() => {
      if (cancelled) return;
      const stored = window.localStorage.getItem("facility-inspection-history");
      if (stored) {
        try {
          setHistory((JSON.parse(stored) as InspectionRecord[]).map((record) => normalizeRecord(record)));
        } catch {
          setHistory([]);
        }
      }
      setHistoryReady(true);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!historyReady) return;
    try {
      window.localStorage.setItem("facility-inspection-history", JSON.stringify(compactHistoryForStorage(history)));
    } catch {
      window.localStorage.removeItem("facility-inspection-history");
    }
  }, [history, historyReady]);

  useEffect(() => {
    if (page !== "monitoring") return;
    let cancelled = false;
    queueMicrotask(() => {
      if (cancelled) return;
      const raw = window.sessionStorage.getItem("facility-focused-inspection");
      if (!raw) return;
      window.sessionStorage.removeItem("facility-focused-inspection");
      try {
        const payload = JSON.parse(raw) as {
          result: InferenceResult;
          imagePreview?: string;
          panel?: MonitorPanel;
          message?: string;
        };
        const restored = normalizeResult(payload.result);
        setWorkspaceView("monitor");
        setLatest(restored);
        setSelectedDetectionId(restored.detections[0]?.id ?? "");
        setSourceType("image");
        setVideoObjectUrl((current) => {
          if (current) URL.revokeObjectURL(current);
          return "";
        });
        setImagePreview(payload.imagePreview ?? restored.evidenceImage ?? restored.annotatedImage ?? "");
        setMonitorPanel(payload.panel ?? "findings");
        setApiMessage(payload.message ?? "Focused inspection loaded from history.");
      } catch {
        setApiMessage("Could not restore the focused inspection from history.");
      }
    });
    return () => {
      cancelled = true;
    };
  }, [page]);

  useEffect(() => {
    if (!analysisStartedAt) {
      return;
    }
    const timer = window.setInterval(() => {
      setAnalysisElapsedSec(Math.max(0, Math.round((Date.now() - analysisStartedAt) / 1000)));
    }, 500);
    return () => window.clearInterval(timer);
  }, [analysisStartedAt]);

  useEffect(() => {
    async function loadApiState() {
      try {
        const [healthResponse, modelResponse] = await Promise.all([
          fetch(`${API_BASE}/health`),
          fetch(`${API_BASE}/api/models`),
        ]);
        if (!healthResponse.ok || !modelResponse.ok) {
          throw new Error("Inference API did not return a healthy response.");
        }
        const modelPayload = (await modelResponse.json()) as { models: ModelOption[] };
        const apiModels = modelPayload.models;
        const availableModels = [
          ...apiModels,
          ...(apiModels.some((item) => item.kind === "detector") ? [] : [fallbackModels[0]]),
          ...(apiModels.some((item) => item.kind === "severity") ? [] : [fallbackModels[1]]),
        ];
        setModels(availableModels);
        const detector = preferredModelValue(availableModels, "detector");
        const severity = preferredModelValue(availableModels, "severity");
        setSettings((current) => ({
          ...current,
          detectorPath: current.detectorPath || detector,
          severityPath: current.severityPath || severity,
        }));
        setApiOnline(true);
        setApiMessage("Inference API connected");
      } catch {
        setApiOnline(false);
        setApiMessage("Inference API offline. Start the FastAPI service to run real inference.");
      }
    }
    loadApiState();
  }, []);

  const stopScanning = useCallback(() => {
    if (intervalRef.current) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setScanning(false);
  }, []);

  const stopCamera = useCallback(() => {
    stopScanning();
    const stream = cameraStreamRef.current ?? (videoRef.current?.srcObject as MediaStream | null);
    stream?.getTracks().forEach((track) => track.stop());
    cameraStreamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    setCameraMessage("Camera preview is off.");
  }, [stopScanning]);

  useEffect(() => {
    return () => {
      stopScanning();
      stopCamera();
    };
  }, [stopCamera, stopScanning]);

  useEffect(() => {
    const video = videoRef.current;
    const stream = cameraStreamRef.current;
    if (!video || !stream || !cameraActive || sourceType !== "camera") return;
    video.srcObject = stream;
    void video.play().catch(() => {
      setCameraMessage("Camera stream is ready, but the browser needs a user gesture to play it.");
    });
  }, [cameraActive, sourceType]);

  const startCamera = async () => {
    setSourceType("camera");
    setImagePreview("");
    setVideoObjectUrl("");
    setLatest(emptyInferenceResult());
    setSelectedDetectionId("");
    setCameraError("");
    setCameraMessage("Requesting camera access...");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      cameraStreamRef.current = stream;
      setCameraActive(true);
      setCameraMessage("Live monitor active. Frames are sampled through both model stages.");
      setApiMessage(apiOnline ? "Camera ready" : apiMessage);
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : "Camera permission was blocked or no camera is available.";
      setCameraActive(false);
      setCameraError(message);
      setCameraMessage("Camera could not be opened.");
      setApiMessage("Camera permission was blocked or no camera is available.");
      return false;
    }
  };

  const applyRecommended = () => {
    setSettings((current) => ({
      ...current,
      ...balancedSettings,
    }));
  };

  const setMode = (mode: Settings["mode"]) => {
    setSettings((current) => ({
      ...current,
      mode,
      ...modeSettings[mode],
    }));
  };

  const captureCurrentFrameBlob = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.videoWidth === 0 || video.videoHeight === 0) return null;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) return null;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return new Promise<Blob | null>((resolve) => {
      canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.86);
    });
  };

  const recordResult = (
    result: InferenceResult,
    source: InspectionRecord["source"],
    preview?: string,
    options?: { sourceUrl?: string; sessionId?: string },
  ) => {
    const normalized = normalizeResult(result);
    const frameSummaries = source === "video" ? defectFrameSummaries(normalized) : [];
    const singleSummary = source === "video" && frameSummaries.length === 0 ? singleFrameSummary(normalized) : null;
    const frames = singleSummary ? [singleSummary] : frameSummaries;
    const recordId = source === "video" ? options?.sessionId || videoSessionIdRef.current || crypto.randomUUID() : crypto.randomUUID();
    const { source: mediaSource, ...recordPayload } = normalized;
    const record: InspectionRecord = {
      ...recordPayload,
      id: recordId,
      createdAt: new Date().toLocaleString(),
      source,
      mediaSource,
      asset: source === "video" ? (options?.sourceUrl ? "Linked Video Inspection" : "Video Inspection") : "Main Building / External Wall",
      preview: normalized.evidenceImage ?? normalized.annotatedImage ?? preview,
      sourceUrl: options?.sourceUrl,
      frames,
      scannedFrames: mediaSource?.sampleCount ?? (source === "video" ? 1 : undefined),
    };
    setHistory((current) => {
      if (source !== "video") return [record, ...current].slice(0, 24);
      const existing = current.find((item) => item.id === recordId);
      if (!existing) return [record, ...current].slice(0, 24);

      const mergedFrames = [...(record.frames ?? []), ...(existing.frames ?? [])]
        .filter((frame, index, all) => all.findIndex((candidate) => candidate.id === frame.id) === index)
        .sort((a, b) => a.frameTimeSec - b.frameTimeSec)
        .slice(0, 120);
      const merged: InspectionRecord = {
        ...existing,
        ...record,
        createdAt: existing.createdAt,
        preview: record.preview ?? existing.preview,
        annotatedImage: record.annotatedImage ?? existing.annotatedImage,
        frames: mergedFrames,
        scannedFrames: Math.max(record.scannedFrames ?? 0, existing.scannedFrames ?? 0),
      };
      return [merged, ...current.filter((item) => item.id !== recordId)].slice(0, 24);
    });
  };

  const inferBlob = async (blob: Blob, source: InspectionRecord["source"], preview?: string) => {
    if (!settings.detectorPath || !settings.severityPath) {
      setApiMessage("Select both detector and severity models before running inference.");
      return;
    }
    if (!apiOnline) {
      setApiMessage("Inference API is offline. Start the FastAPI service, then refresh.");
      return;
    }
    const formData = new FormData();
    formData.append("file", blob, `${source}_frame.jpg`);
    formData.append("detector_path", settings.detectorPath);
    formData.append("severity_path", settings.severityPath);
    formData.append("confidence", String(settings.confidence));
    formData.append("iou", String(settings.iou));

    setBusy(true);
    try {
      const response = await fetch(`${API_BASE}/api/infer/image`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = normalizeResult((await response.json()) as InferenceResult);
      setLatest(payload);
      setSelectedDetectionId(payload.detections[0]?.id ?? "");
      setMonitorPanel("findings");
      recordResult(payload, source, payload.evidenceImage ?? payload.annotatedImage ?? preview, {
        sessionId: source === "video" ? videoSessionIdRef.current : undefined,
      });
      setApiMessage(`Inference complete in ${formatMs(payload.latencyMs)}`);
    } catch {
      setApiMessage("Inference failed. Check model paths and backend logs.");
    } finally {
      setBusy(false);
    }
  };

  const inferLinkedMedia = async () => {
    const trimmedUrl = mediaUrl.trim();
    if (!trimmedUrl) {
      setApiMessage("Paste a public Drive, image/video, or YouTube URL first.");
      return;
    }
    if (!settings.detectorPath || !settings.severityPath) {
      setApiMessage("Select both detector and severity models before running URL inference.");
      return;
    }
    if (!apiOnline) {
      setApiMessage("Inference API is offline. Start the FastAPI service, then refresh.");
      return;
    }

    const formData = new FormData();
    formData.append("media_url", trimmedUrl);
    formData.append("detector_path", settings.detectorPath);
    formData.append("severity_path", settings.severityPath);
    formData.append("confidence", String(settings.confidence));
    formData.append("iou", String(settings.iou));
    formData.append("frame_time_sec", String(linkFrameTime));
    formData.append("scan_video", String(scanLinkedVideo));
    formData.append("scan_fps", String(linkScanFps));
    formData.append("review_confidence", String(Math.min(settings.confidence, 0.18)));

    const linkedSource: InspectionRecord["source"] = /youtube\.com|youtu\.be|\.mp4|\.mov|\.webm|\.avi/i.test(trimmedUrl)
      ? "video"
      : "image";
    const queueAsVideoJob = linkedSource === "video" && scanLinkedVideo;

    if (queueAsVideoJob) {
      if (activeVideoJob) {
        setApiMessage("A video analysis task is already queued or running. Wait for it to finish or cancel it first.");
        setMonitorPanel("frames");
        return;
      }
      try {
        const response = await fetch(`${API_BASE}/api/jobs/video`, {
          method: "POST",
          body: formData,
        });
        if (!response.ok) {
          const detail = await response.text();
          throw new Error(detail);
        }
        const job = (await response.json()) as VideoJob;
        setVideoJob(job);
        setMonitorPanel("frames");
        setApiMessage(`Video analysis queued as task ${job.id.slice(0, 8)}. You can keep using live or image inference.`);
      } catch (error) {
        const message = error instanceof Error ? error.message : "Video task submission failed.";
        setApiMessage(message.slice(0, 180));
      }
      return;
    }

    stopCamera();
    stopScanning();
    setSourceType("image");
    setImagePreview("");
    setVideoObjectUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return "";
    });
    setLatest(emptyInferenceResult());
    setSelectedDetectionId("");

    setBusy(true);
    setAnalysisElapsedSec(0);
    setAnalysisStartedAt(Date.now());
    setApiMessage(scanLinkedVideo ? `Scanning linked media at ${linkScanFps} FPS...` : "Fetching linked media and running inference...");
    try {
      const response = await fetch(`${API_BASE}/api/infer/url`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail);
      }
      const payload = normalizeResult((await response.json()) as InferenceResult);
      setLatest(payload);
      setSelectedDetectionId(payload.detections[0]?.id ?? "");
      const linkedSessionId = linkedSource === "video" ? crypto.randomUUID() : undefined;
      if (linkedSessionId) {
        videoSessionIdRef.current = linkedSessionId;
      }
      setMonitorPanel(linkedSource === "video" ? "frames" : "findings");
      recordResult(payload, linkedSource, payload.evidenceImage ?? payload.annotatedImage, {
        sourceUrl: trimmedUrl,
        sessionId: linkedSessionId,
      });
      const frameNote =
        payload.source?.selectedFrameTimeSec !== undefined ? ` at ${payload.source.selectedFrameTimeSec.toFixed(1)}s` : "";
      const fpsNote = payload.source?.scanFps ? ` (${payload.source.sampleCount ?? 1} frames @ ${payload.source.scanFps} FPS)` : "";
      const defectFrameNote =
        payload.source?.positiveFrameCount !== undefined ? `, ${payload.source.positiveFrameCount} defect-positive frames` : "";
      const capNote = payload.source?.safetyCapped ? " capped by runtime safety limit" : "";
      const reviewNote = payload.source?.reviewMode ? ` using review sensitivity ${payload.source.usedConfidence}` : "";
      setApiMessage(`Linked media scan complete${frameNote}${fpsNote}${defectFrameNote}${capNote}${reviewNote} in ${formatMs(payload.latencyMs)}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Linked media inference failed.";
      setApiMessage(message.slice(0, 180));
    } finally {
      setBusy(false);
      setAnalysisStartedAt(null);
    }
  };

  const runCurrentFrameOnce = async () => {
    const blob = await captureCurrentFrameBlob();
    if (blob) {
      await inferBlob(blob, sourceType === "video" ? "video" : "camera");
    } else {
      setApiMessage(sourceType === "video" ? "Video frame is not ready yet." : "Camera frame is not ready yet.");
    }
  };

  const startSamplingLoop = () => {
    if (intervalRef.current) {
      window.clearInterval(intervalRef.current);
    }
    window.setTimeout(() => {
      void runCurrentFrameOnce();
    }, 500);
    intervalRef.current = window.setInterval(() => {
      void runCurrentFrameOnce();
    }, settings.frameIntervalMs);
    setScanning(true);
  };

  const toggleLiveMonitoring = async () => {
    if (scanning || (sourceType !== "video" && cameraActive)) {
      if (sourceType === "video") {
        stopScanning();
      } else {
        stopCamera();
      }
      return;
    }

    if (sourceType === "video") {
      startSamplingLoop();
      setApiMessage("Video scan active. The visible frame is sampled repeatedly.");
      return;
    }

    const opened = await startCamera();
    if (opened) {
      startSamplingLoop();
    }
  };

  const onUploadMedia = async (file: File | undefined) => {
    if (!file) return;
    stopCamera();
    stopScanning();
    setLatest(emptyInferenceResult());
    setSelectedDetectionId("");

    const preview = URL.createObjectURL(file);
    if (file.type.startsWith("video/")) {
      videoSessionIdRef.current = crypto.randomUUID();
      setSourceType("video");
      setImagePreview("");
      setVideoObjectUrl((current) => {
        if (current) URL.revokeObjectURL(current);
        return preview;
      });
      setCameraMessage("Video evidence loaded. Use Analyze frame once, or Scan video to sample visible playback frames.");
      setApiMessage("Video loaded. Sampling uses the visible frame in the center evidence viewport.");
      return;
    }

    setSourceType("image");
    setVideoObjectUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return "";
    });
    setImagePreview(preview);
    await inferBlob(file, "image", preview);
  };

  useEffect(() => {
    uploadMediaRef.current = onUploadMedia;
  });

  const isFileDrag = (event: DragEvent<HTMLElement>) => Array.from(event.dataTransfer.types).includes("Files");

  const onDragEnterMedia = (event: DragEvent<HTMLElement>) => {
    if (!isFileDrag(event)) return;
    event.preventDefault();
    event.stopPropagation();
    setDragActive(true);
  };

  const onDragOverMedia = (event: DragEvent<HTMLElement>) => {
    if (!isFileDrag(event)) return;
    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect = "copy";
    setDragActive(true);
  };

  const onDragLeaveMedia = (event: DragEvent<HTMLElement>) => {
    event.stopPropagation();
    if (event.currentTarget === event.target) {
      setDragActive(false);
    }
  };

  const onDropMedia = async (event: DragEvent<HTMLElement>) => {
    if (!isFileDrag(event)) return;
    event.preventDefault();
    event.stopPropagation();
    setDragActive(false);
    const file = event.dataTransfer.files?.[0];
    if (!file) return;
    setWorkspaceView("monitor");
    await onUploadMedia(file);
  };

  useEffect(() => {
    const hasFiles = (event: globalThis.DragEvent) => Array.from(event.dataTransfer?.types ?? []).includes("Files");
    const handleDragEnter = (event: globalThis.DragEvent) => {
      if (!hasFiles(event)) return;
      event.preventDefault();
      setDragActive(true);
    };
    const handleDragOver = (event: globalThis.DragEvent) => {
      if (!hasFiles(event)) return;
      event.preventDefault();
      if (event.dataTransfer) {
        event.dataTransfer.dropEffect = "copy";
      }
      setDragActive(true);
    };
    const handleDragLeave = (event: globalThis.DragEvent) => {
      if (!event.relatedTarget) {
        setDragActive(false);
      }
    };
    const handleDrop = (event: globalThis.DragEvent) => {
      if (!hasFiles(event)) return;
      event.preventDefault();
      setDragActive(false);
      const file = event.dataTransfer?.files?.[0];
      if (!file) return;
      setWorkspaceView("monitor");
      void uploadMediaRef.current?.(file);
    };
    window.addEventListener("dragenter", handleDragEnter);
    window.addEventListener("dragover", handleDragOver);
    window.addEventListener("dragleave", handleDragLeave);
    window.addEventListener("drop", handleDrop);
    return () => {
      window.removeEventListener("dragenter", handleDragEnter);
      window.removeEventListener("dragover", handleDragOver);
      window.removeEventListener("dragleave", handleDragLeave);
      window.removeEventListener("drop", handleDrop);
    };
  }, []);

  const restoreVideoFrame = (frame: VideoFrameSummary, record?: InspectionRecord) => {
    stopScanning();
    if (cameraActive) {
      stopCamera();
    }
    const base = record ? recordToInferenceResult(record) : latest;
    const frameDetections = frame.detectionItems && frame.detectionItems.length > 0 ? frame.detectionItems : base.detections;
    const restored = normalizeResult({
      ...base,
      detections: frameDetections,
      conditionIndex: frame.conditionIndex ?? base.conditionIndex,
      riskScore: frame.riskScore ?? base.riskScore,
      evidenceImage: frame.evidenceImage ?? base.evidenceImage ?? record?.preview,
      annotatedImage: frame.annotatedImage ?? base.annotatedImage,
      source: {
        ...(base.source ?? {}),
        selectedFrameTimeSec: frame.frameTimeSec,
        frameTimeSec: frame.frameTimeSec,
      },
    });
    setWorkspaceView("monitor");
    setLatest(restored);
    setSelectedDetectionId(restored.detections[0]?.id ?? "");
    setSourceType("image");
    setVideoObjectUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return "";
    });
    setImagePreview(restored.evidenceImage ?? restored.annotatedImage ?? record?.preview ?? "");
    setApiMessage(
      `Restored video frame ${frame.frameTimeSec.toFixed(1)}s with ${frameDetections.length} detected defect${
        frameDetections.length === 1 ? "" : "s"
      }.`,
    );
    const targetPanel: MonitorPanel = frameDetections.length ? "selected" : "findings";
    setMonitorPanel(targetPanel);
    if (page !== "monitoring") {
      window.sessionStorage.setItem(
        "facility-focused-inspection",
        JSON.stringify({
          result: restored,
          imagePreview: restored.evidenceImage ?? restored.annotatedImage ?? record?.preview ?? "",
          panel: targetPanel,
          message: `Focused video frame ${frame.frameTimeSec.toFixed(1)}s loaded from history.`,
        }),
      );
      router.push("/monitoring");
    }
  };

  const restoreHistoryRecord = (record: InspectionRecord) => {
    stopScanning();
    if (cameraActive) {
      stopCamera();
    }
    const restored = recordToInferenceResult(record);
    setLatest(restored);
    setSelectedDetectionId(restored.detections[0]?.id ?? "");
    setSourceType("image");
    setVideoObjectUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return "";
    });
    const preview = restored.evidenceImage ?? restored.annotatedImage ?? record.preview ?? "";
    const targetPanel: MonitorPanel =
      record.source === "video" && record.frames?.length ? "frames" : restored.detections.length ? "selected" : "findings";
    const message = `Focused ${record.source} inspection loaded from history.`;
    setImagePreview(preview);
    setApiMessage(message);
    setMonitorPanel(targetPanel);
    setWorkspaceView("monitor");
    if (page !== "monitoring") {
      window.sessionStorage.setItem(
        "facility-focused-inspection",
        JSON.stringify({
          result: restored,
          imagePreview: preview,
          panel: targetPanel,
          message,
        }),
      );
      router.push("/monitoring");
    }
  };

  const selectDetection = (id: string) => {
    setSelectedDetectionId(id);
    setMonitorPanel("selected");
  };

  const loadVideoJobResult = (job: VideoJob) => {
    if (!job.result) {
      setApiMessage("Video task has no completed result to load yet.");
      return;
    }
    const payload = normalizeResult(job.result);
    stopScanning();
    if (cameraActive) {
      stopCamera();
    }
    setWorkspaceView("monitor");
    setLatest(payload);
    setSelectedDetectionId(payload.detections[0]?.id ?? "");
    setSourceType("image");
    setVideoObjectUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return "";
    });
    setImagePreview(payload.evidenceImage ?? payload.annotatedImage ?? "");
    setMonitorPanel("frames");
    recordResult(payload, "video", payload.evidenceImage ?? payload.annotatedImage, {
      sourceUrl: job.mediaUrl,
      sessionId: job.id,
    });
    setApiMessage(
      `Loaded video task result: ${payload.detections.length} finding${payload.detections.length === 1 ? "" : "s"} across ${
        payload.source?.positiveFrameCount ?? 0
      } defect-positive frame${(payload.source?.positiveFrameCount ?? 0) === 1 ? "" : "s"}.`,
    );
  };

  const cancelVideoJob = async () => {
    if (!videoJob || !activeVideoJob) return;
    try {
      const response = await fetch(`${API_BASE}/api/jobs/${videoJob.id}`, { method: "DELETE" });
      if (!response.ok) throw new Error(await response.text());
      const job = (await response.json()) as VideoJob;
      setVideoJob(job);
      setApiMessage(job.message || "Video task cancellation requested.");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not cancel video task.";
      setApiMessage(message.slice(0, 180));
    }
  };

  useEffect(() => {
    if (!videoJobId || !activeVideoJob || !apiOnline) return;

    let cancelled = false;
    const pollJob = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/jobs/${videoJobId}`);
        if (!response.ok) throw new Error(await response.text());
        const job = (await response.json()) as VideoJob;
        if (cancelled) return;
        setVideoJob(job);
        if (job.status === "completed" && !completedVideoJobIdsRef.current.has(job.id)) {
          completedVideoJobIdsRef.current.add(job.id);
          setApiMessage("Video analysis task completed. Load the result when you are ready.");
          setMonitorPanel("frames");
        } else if (job.status === "failed") {
          setApiMessage((job.error || job.message || "Video analysis task failed.").slice(0, 180));
        } else if (job.status === "cancelled") {
          setApiMessage("Video analysis task cancelled.");
        }
      } catch (error) {
        if (!cancelled) {
          const message = error instanceof Error ? error.message : "Could not poll video analysis task.";
          setApiMessage(message.slice(0, 180));
        }
      }
    };

    void pollJob();
    const timer = window.setInterval(() => {
      void pollJob();
    }, 1200);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [videoJobId, activeVideoJob, apiOnline]);

  const analysisActive = busy && Boolean(analysisStartedAt);
  const liveMediaVisible = (sourceType === "camera" && cameraActive) || (sourceType === "video" && Boolean(videoObjectUrl));
  const cleanEvidenceImage = liveMediaVisible ? "" : latest.evidenceImage || (sourceType === "image" && !latest.annotatedImage ? imagePreview : "");
  const evidenceImage = liveMediaVisible ? "" : cleanEvidenceImage || latest.annotatedImage || (sourceType === "image" ? imagePreview : "");
  const showVideo = liveMediaVisible;
  const showInteractiveDetections = showVideo || Boolean(cleanEvidenceImage);
  const monitorReviewMode =
    workspaceView === "monitor" &&
    (cameraActive || scanning || latest.latencyMs > 0 || Boolean(evidenceImage) || Boolean(videoObjectUrl));
  const monitorFixed = workspaceView === "monitor" && !monitorReviewMode;
  const zoomPercent = Math.round(evidenceZoom * 100);
  const stageTotalMs = latest.stageMetrics.stage1LatencyMs + latest.stageMetrics.stage2LatencyMs;
  const estimatedThroughput = stageTotalMs > 0 ? `${(1000 / Math.max(stageTotalMs, 1)).toFixed(1)} fps` : "n/a";
  const queueStatus = videoJob
    ? `${videoJob.status}${typeof videoJob.progress === "number" ? ` ${Math.round(videoJob.progress)}%` : ""}`
    : "idle";
  const frameCadence = scanning || cameraActive ? `${settings.frameIntervalMs} ms` : latest.source?.scanFps ? `${latest.source.scanFps} fps` : "idle";

  return (
    <main
      className={`dashboard-shell enterprise-shell bg-[var(--app-bg)] text-[var(--app-text)] ${
        monitorFixed ? "monitor-fit h-screen overflow-hidden" : "min-h-screen overflow-x-hidden"
      }`}
      style={themeStyle}
      onDragEnter={onDragEnterMedia}
      onDragOver={onDragOverMedia}
      onDragLeave={onDragLeaveMedia}
      onDrop={onDropMedia}
    >
      <div
        className={`grid ${sidebarCollapsed ? "grid-cols-[76px_minmax(0,1fr)]" : "grid-cols-[260px_minmax(0,1fr)]"} max-md:grid-cols-1 ${
          monitorFixed ? "h-screen min-h-0 overflow-hidden" : "min-h-screen"
        }`}
      >
        <aside className="enterprise-sidebar hidden border-r border-white/10 bg-[#11161d] md:block">
          <div className={`border-b border-white/10 ${sidebarCollapsed ? "px-2 py-4" : "px-5 py-5"}`}>
            <div className={`flex items-center ${sidebarCollapsed ? "justify-center" : "gap-3"}`}>
              <button
                className="grid h-11 w-11 place-items-center border border-cyan-300/30 bg-cyan-300/10 text-cyan-200 transition hover:bg-cyan-300/15"
                onClick={() => setSidebarCollapsed((value) => !value)}
                title={sidebarCollapsed ? "Expand navigation" : "Collapse navigation"}
                aria-label={sidebarCollapsed ? "Expand navigation" : "Collapse navigation"}
              >
                <LayoutDashboard className="h-5 w-5 text-cyan-200" />
              </button>
              {!sidebarCollapsed ? (
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-cyan-200">Facility AI</p>
                <p className="mt-1 text-sm text-slate-400">Inspection Ops</p>
              </div>
              ) : null}
            </div>
          </div>
          <nav className={`flex flex-col gap-1 ${sidebarCollapsed ? "px-2 py-4" : "px-3 py-4"}`}>
            {navItems.map(({ label, icon: Icon, page: itemPage, href }) => {
              const active = itemPage === page;
              return (
                <Link
                  key={label}
                  href={href}
                  className={`flex h-12 w-full items-center border-l-2 text-sm font-semibold transition ${
                    active
                      ? "border-cyan-300 bg-cyan-300/[0.12] text-cyan-100"
                      : "border-transparent text-slate-300 hover:border-cyan-300/40 hover:bg-white/[0.04] hover:text-cyan-100"
                  } ${sidebarCollapsed ? "justify-center px-0" : "gap-3 px-3"}`}
                  title={label}
                  aria-current={active ? "page" : undefined}
                >
                  <Icon className="h-5 w-5" />
                  {!sidebarCollapsed ? <span>{label}</span> : null}
                </Link>
              );
            })}
          </nav>
          {!sidebarCollapsed ? (
          <div className="mx-5 mt-4 border-t border-white/10 pt-4">
            <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Deployment</p>
            <p className="mt-2 text-sm font-semibold text-slate-100">Two-stage YOLO</p>
            <p className="mt-1 text-xs leading-5 text-slate-400">Detection, crop severity, evidence, and inspection history.</p>
          </div>
          ) : null}
        </aside>

        <section
          className={`grid grid-rows-[auto_minmax(0,1fr)] ${
            monitorFixed ? "h-screen min-h-0 overflow-hidden" : "min-h-screen"
          }`}
        >
          <header className="enterprise-header border-b border-white/10 bg-[#11161d]/95 px-6 py-5 backdrop-blur">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.25em] text-cyan-200">AIENG / {currentPageMeta.eyebrow}</p>
                <h1 className="mt-1 text-3xl font-semibold tracking-normal text-white">{currentPageMeta.title}</h1>
                <p className={`mt-1 max-w-2xl text-sm text-slate-400 ${workspaceView === "monitor" ? "hidden 2xl:block" : ""}`}>
                  {currentPageMeta.subtitle}
                </p>
              </div>

              {workspaceView === "monitor" ? (
                <div className="flex items-center justify-end">
                  <button
                    className="theme-ghost-button inline-flex h-10 items-center justify-center gap-2 rounded-lg border border-white/10 bg-white/[0.04] px-4 text-sm font-semibold text-slate-100 transition hover:bg-white/[0.08]"
                    onClick={() => setSettingsPanelOpen(true)}
                    aria-haspopup="dialog"
                  >
                    <Settings2 className="h-4 w-4" />
                    Settings
                  </button>
                </div>
              ) : (
                <div className="flex flex-wrap items-center gap-3">
                  <button
                    className="theme-ghost-button inline-flex h-11 items-center justify-center gap-2 rounded-lg border border-white/10 bg-white/[0.04] px-4 text-sm font-semibold text-slate-100 transition hover:bg-white/[0.08]"
                    onClick={() => setThemeMode((value) => (value === "dark" ? "light" : "dark"))}
                  >
                    {themeMode === "dark" ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
                    {themeMode === "dark" ? "Dark" : "Light"}
                  </button>
                  <Link
                    href="/monitoring"
                    className="theme-primary-button inline-flex h-11 items-center justify-center gap-2 rounded-lg bg-teal-300 px-4 text-sm font-semibold text-slate-950 transition hover:bg-teal-200"
                  >
                    <Camera className="h-4 w-4" />
                    Open Monitoring
                  </Link>
                </div>
              )}
            </div>

          </header>

          {workspaceView === "monitor" && settingsPanelOpen ? (
            <div className="fixed inset-0 z-50 flex justify-end bg-black/55 p-3 backdrop-blur-sm" role="dialog" aria-modal="true">
              <div className="enterprise-panel flex max-h-[calc(100vh-1.5rem)] w-full max-w-[440px] flex-col overflow-hidden border border-white/10 bg-[#111820] shadow-2xl">
                <div className="flex items-center justify-between border-b border-white/10 p-4">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-200">Inference Controls</p>
                    <h2 className="mt-1 text-lg font-semibold text-white">Settings</h2>
                  </div>
                  <button
                    className="theme-ghost-button grid h-9 w-9 place-items-center border border-white/10 bg-white/[0.04] text-slate-100"
                    onClick={() => setSettingsPanelOpen(false)}
                    aria-label="Close settings"
                  >
                    <ChevronRight className="h-4 w-4" />
                  </button>
                </div>

                <div className="min-h-0 flex-1 space-y-4 overflow-auto p-4">
                  <label className="block text-sm text-slate-300">
                    <span className="mb-2 block text-xs uppercase tracking-[0.16em] text-slate-500">Detector Model</span>
                    <select
                      className="h-10 w-full min-w-0 rounded-lg border border-white/10 bg-[#0d1218] px-3 text-sm text-slate-100 outline-none focus:border-cyan-300/60"
                      value={settings.detectorPath}
                      onChange={(event) => setSettings((current) => ({ ...current, detectorPath: event.target.value }))}
                    >
                      {detectorModels.map((model) => (
                        <option key={model.value || model.label} value={model.value}>
                          {model.label}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="block text-sm text-slate-300">
                    <span className="mb-2 block text-xs uppercase tracking-[0.16em] text-slate-500">Severity Model</span>
                    <select
                      className="h-10 w-full min-w-0 rounded-lg border border-white/10 bg-[#0d1218] px-3 text-sm text-slate-100 outline-none focus:border-cyan-300/60"
                      value={settings.severityPath}
                      onChange={(event) => setSettings((current) => ({ ...current, severityPath: event.target.value }))}
                    >
                      {severityModels.map((model) => (
                        <option key={model.value || model.label} value={model.value}>
                          {model.label}
                        </option>
                      ))}
                    </select>
                  </label>

                  <div className="grid grid-cols-2 gap-3">
                    <button
                      className="theme-ghost-button inline-flex h-10 items-center justify-center gap-2 border border-white/10 bg-white/[0.04] px-3 text-sm font-semibold text-slate-100"
                      onClick={() => setThemeMode((value) => (value === "dark" ? "light" : "dark"))}
                    >
                      {themeMode === "dark" ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
                      {themeMode === "dark" ? "Dark" : "Light"}
                    </button>

                    <label className="relative">
                      <Palette className="pointer-events-none absolute left-3 top-3 h-4 w-4 text-slate-400" />
                      <select
                        className="h-10 w-full rounded-lg border border-white/10 bg-[#0d1218] pl-9 pr-3 text-sm text-slate-100 outline-none focus:border-cyan-300/60"
                        value={paletteName}
                        onChange={(event) => setPaletteName(event.target.value as PaletteName)}
                      >
                        {(Object.keys(paletteThemes) as PaletteName[]).map((name) => (
                          <option key={name}>{name}</option>
                        ))}
                      </select>
                    </label>
                  </div>

                  <label className="block text-sm text-slate-300">
                    <span className="mb-2 block text-xs uppercase tracking-[0.16em] text-slate-500">Mode</span>
                    <select
                      className="h-10 w-full rounded-lg border border-white/10 bg-[#111820] px-3 text-slate-100 outline-none"
                      value={settings.mode}
                      onChange={(event) => setMode(event.target.value as Settings["mode"])}
                    >
                      <option>Near real-time</option>
                      <option>Balanced</option>
                      <option>High accuracy</option>
                      <option>Multi-defect review</option>
                    </select>
                    {settings.mode === "Multi-defect review" ? (
                      <span className="mt-2 block rounded-lg border border-amber-300/25 bg-amber-300/10 px-3 py-2 text-xs leading-5 text-amber-50/80">
                        Uses confidence 0.10 and IoU 0.55 to reveal multiple separated candidates in crowded frames. Expect more review candidates and possible false positives.
                      </span>
                    ) : null}
                  </label>

                  <Slider
                    label="Confidence"
                    value={settings.confidence}
                    min={0.05}
                    max={0.95}
                    step={0.05}
                    onChange={(value) => setSettings((current) => ({ ...current, confidence: value }))}
                  />
                  <Slider
                    label="NMS IoU"
                    value={settings.iou}
                    min={0.1}
                    max={0.9}
                    step={0.05}
                    onChange={(value) => setSettings((current) => ({ ...current, iou: value }))}
                  />
                  <Slider
                    label="Frame interval"
                    suffix="ms"
                    value={settings.frameIntervalMs}
                    min={300}
                    max={2200}
                    step={100}
                    onChange={(value) => setSettings((current) => ({ ...current, frameIntervalMs: value }))}
                  />
                </div>

                <div className="grid grid-cols-2 gap-3 border-t border-white/10 p-4">
                  <button
                    className="theme-primary-button inline-flex h-10 items-center justify-center gap-2 bg-teal-300 px-3 text-sm font-semibold text-slate-950"
                    onClick={applyRecommended}
                  >
                    <Zap className="h-4 w-4" />
                    Recommended
                  </button>
                  <button
                    className="theme-ghost-button inline-flex h-10 items-center justify-center gap-2 border border-white/10 bg-white/[0.04] px-3 text-sm font-semibold text-slate-100"
                    onClick={() => setSettingsPanelOpen(false)}
                  >
                    Done
                  </button>
                </div>
              </div>
            </div>
          ) : null}

          {workspaceView === "overview" ? (
            historyMode ? (
              <div className="grid min-h-0 gap-5 p-5 xl:grid-cols-[minmax(440px,0.95fr)_minmax(0,1.25fr)]">
                <section className="flex min-h-0 flex-col gap-5">
                  <div className="enterprise-panel border border-cyan-300/20 bg-cyan-300/[0.06] p-4">
                    <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-cyan-100">
                      <History className="h-4 w-4" />
                      Evidence archive workflow
                    </div>
                    <p className="text-sm leading-6 text-slate-300">
                      History is for finding saved evidence. Click{" "}
                      <strong className="text-white">Open result for focused review</strong> to move one inspection into
                      Monitoring, where you can inspect boxes, severity, recommended actions, and frame evidence.
                    </p>
                  </div>
                  <div ref={historyRef}>
                    <HistoryPanel
                      history={history}
                      expandedHistoryId={expandedHistoryId}
                      onToggleExpanded={(id) => setExpandedHistoryId((current) => (current === id ? "" : id))}
                      onRestoreRecord={restoreHistoryRecord}
                      onRestoreFrame={restoreVideoFrame}
                      large
                    />
                  </div>
                </section>
                <section className="grid min-h-0 gap-5 lg:grid-cols-2 xl:grid-cols-1">
                  <HistoryVisualGallery
                    history={history}
                    onRestoreRecord={restoreHistoryRecord}
                    onRestoreFrame={restoreVideoFrame}
                  />
                  <ArchiveSummaryPanel
                    records={history.length}
                    detections={analyticsDetections.length}
                    critical={criticalCount}
                    immediate={immediateCount}
                    condition={signalCondition}
                  />
                  <div className="enterprise-panel border border-white/10 bg-[#111820] p-4">
                    <div className="mb-3 flex items-center justify-between">
                      <h2 className="text-sm font-semibold text-white">Archive Use</h2>
                      <ShieldAlert className="h-4 w-4 text-slate-400" />
                    </div>
                    <div className="space-y-3 text-sm leading-6 text-slate-300">
                      <p>
                        Use this page when you need to reopen evidence from a previous image, video, or live scan.
                      </p>
                      <p>
                        Use Analytics when you need aggregate trends across records, not a single inspection review.
                      </p>
                    </div>
                  </div>
                </section>
              </div>
            ) : dashboardMode ? (
              <div className="grid min-h-0 gap-5 p-5 xl:grid-cols-[minmax(0,1.35fr)_minmax(340px,0.75fr)]">
                <section className="flex min-h-0 flex-col gap-5">
                  <DashboardOpsPalette
                    history={history}
                    apiOnline={apiOnline}
                    busy={busy}
                    scanning={scanning}
                    activeVideoJob={activeVideoJob}
                    modelCount={models.filter((model) => model.value).length}
                    estimatedThroughput={estimatedThroughput}
                    frameCadence={frameCadence}
                    queueStatus={queueStatus}
                  />
                  <DashboardOperationsOverview
                    history={history}
                    videoJob={videoJob}
                    apiOnline={apiOnline}
                    apiMessage={apiMessage}
                    latest={latest}
                    estimatedThroughput={estimatedThroughput}
                    frameCadence={frameCadence}
                    queueStatus={queueStatus}
                    onRestoreRecord={restoreHistoryRecord}
                    onRestoreFrame={restoreVideoFrame}
                  />
                </section>

                <section className="flex min-h-0 flex-col gap-5">
                  <div className="enterprise-panel border border-white/10 bg-[#111820] p-4">
                    <div className="mb-3 flex items-center justify-between">
                      <h2 className="text-sm font-semibold text-white">Start Work</h2>
                      <Camera className="h-4 w-4 text-slate-400" />
                    </div>
                    <p className="text-sm leading-6 text-slate-400">
                      Use Monitoring for a new camera, image, video, YouTube, or Drive inspection. Use History only when you need to reopen saved evidence.
                    </p>
                    <div className="mt-4 grid gap-2">
                      <Link
                        href="/monitoring"
                        className="theme-primary-button inline-flex h-11 items-center justify-center gap-2 rounded-lg px-4 text-sm font-semibold text-slate-950"
                      >
                        <Camera className="h-4 w-4" />
                        Open Monitoring
                      </Link>
                      <Link
                        href="/history"
                        className="theme-ghost-button inline-flex h-11 items-center justify-center gap-2 rounded-lg border border-white/10 bg-white/[0.04] px-4 text-sm font-semibold text-slate-100"
                      >
                        <History className="h-4 w-4" />
                        Browse Evidence
                      </Link>
                    </div>
                  </div>
                  <StatusPanel apiOnline={apiOnline} apiMessage={apiMessage} busy={busy} scanning={scanning} />
                  <PipelinePanel
                    stage1Latency={latest.stageMetrics.stage1LatencyMs}
                    stage2Latency={latest.stageMetrics.stage2LatencyMs}
                    cropsClassified={latest.stageMetrics.cropsClassified}
                    detectorModel={
                      latest.stageMetrics.detectorModel === "not selected"
                        ? modelDisplayName(settings.detectorPath)
                        : latest.stageMetrics.detectorModel
                    }
                    severityModel={
                      latest.stageMetrics.severityModel === "not selected"
                        ? modelDisplayName(settings.severityPath)
                        : latest.stageMetrics.severityModel
                    }
                  />
                </section>
              </div>
            ) : (
            <div
              className={`grid min-h-0 gap-5 p-5 ${
                analyticsMode
                  ? "xl:grid-cols-1"
                  : "xl:grid-cols-[minmax(0,1.4fr)_minmax(360px,0.8fr)]"
              }`}
            >
              <section className="flex min-h-0 flex-col gap-5">
                <MetricPalette
                  detections={chartDetections.length}
                  critical={criticalCount}
                  immediate={immediateCount}
                  latency={signalLatency}
                  condition={signalCondition}
                />
                <div className="grid gap-5 2xl:grid-cols-[minmax(0,1.35fr)_minmax(320px,0.65fr)]">
                  <div ref={analyticsRef} className="enterprise-panel relative border border-white/10 bg-[#111820] p-4">
                    <div className="mb-3 flex items-center justify-between">
                      <div>
                        <h2 className="text-sm font-semibold text-white">Condition and Risk Trend</h2>
                        <p className="mt-1 text-xs text-slate-400">
                          A triage trend for inspection records, not a structural certificate. Use it to spot whether asset condition is improving or worsening.
                        </p>
                      </div>
                      <div className="flex shrink-0 items-center gap-2">
                        <button
                          type="button"
                          onClick={() => setRiskInfoOpen((value) => !value)}
                          className="grid h-8 w-8 place-items-center rounded-lg border border-white/10 bg-white/[0.04] text-slate-300 transition hover:bg-white/[0.08]"
                          title="Explain condition and risk index"
                          aria-label="Explain condition and risk index"
                        >
                          <InfoIcon className="h-4 w-4" />
                        </button>
                        <Activity className="h-4 w-4 text-slate-400" />
                      </div>
                    </div>
                    {riskInfoOpen ? (
                      <div className="absolute right-4 top-14 z-20 w-[min(420px,calc(100%-2rem))] rounded-xl border border-white/15 bg-[#090f14]/95 p-4 text-xs shadow-2xl shadow-black/40 backdrop-blur">
                        <div className="mb-3 flex items-center justify-between">
                          <p className="font-semibold text-white">Metric Definitions</p>
                          <button
                            type="button"
                            onClick={() => setRiskInfoOpen(false)}
                            className="rounded-md border border-white/10 px-2 py-1 text-slate-300 hover:bg-white/[0.08]"
                          >
                            Close
                          </button>
                        </div>
                        <div className="grid gap-3">
                          <p className="leading-relaxed text-slate-300">
                            <strong className="text-teal-100">Condition index</strong> is estimated asset health on a 0-100 scale. 100 means no detected issue;
                            lower values mean the evidence contains more serious, larger, or more confident defects.
                          </p>
                          <p className="leading-relaxed text-slate-300">
                            <strong className="text-amber-100">Risk score</strong> is maintenance urgency on a 0-100 scale. It increases with severity tier,
                            detector confidence, defect area, and detection count.
                          </p>
                          <p className="leading-relaxed text-slate-300">
                            The backend computes <strong className="text-white">Condition = 100 - Risk</strong>. A rising risk line or falling condition line means
                            the asset should move higher in the inspection queue.
                          </p>
                        </div>
                      </div>
                    ) : null}
                    <div className="mb-2 flex flex-wrap gap-3 text-xs text-slate-300">
                      <span className="inline-flex items-center gap-2">
                        <span className="h-2 w-6 rounded-full bg-teal-300" />
                        Condition
                      </span>
                      <span className="inline-flex items-center gap-2">
                        <span className="h-2 w-6 rounded-full bg-amber-500" />
                        Risk
                      </span>
                    </div>
                    <div className={analyticsMode ? "h-[420px]" : "h-[340px]"}>
                      {trendData.length ? (
                        <ConditionTrendChart data={trendData} />
                      ) : (
                        <EmptyMini label="Condition trend appears after saved inspection records" />
                      )}
                    </div>
                  </div>

                  <div className="enterprise-panel border border-white/10 bg-[#111820] p-4">
                    <div className="mb-3 flex items-center justify-between">
                      <h2 className="text-sm font-semibold text-white">Severity Mix</h2>
                      <Layers className="h-4 w-4 text-slate-400" />
                    </div>
                    <div className={analyticsMode ? "h-[420px]" : "h-[360px]"}>
                      {severityData.length ? <SeverityPieChart data={severityData} /> : <EmptyMini label="No severity data yet" />}
                    </div>
                  </div>
                </div>
                <div className="enterprise-panel border border-white/10 bg-[#111820] p-4">
                  <div className="mb-3 flex items-center justify-between">
                    <div>
                      <h2 className="text-sm font-semibold text-white">Defect Distribution</h2>
                      <p className="mt-1 text-xs text-slate-400">
                        {aggregateOverviewMode
                          ? "Class balance across saved inspection records."
                          : "Class balance in the latest restored or live inspection result."}
                      </p>
                    </div>
                    <RefreshCcw className="h-4 w-4 text-slate-400" />
                  </div>
                  <div className={analyticsMode ? "h-[340px]" : "h-[280px]"}>
                    {defectData.length ? <DefectBarChart data={defectData} /> : <EmptyMini label="Distribution appears after inference" />}
                  </div>
                </div>
                {analyticsMode ? (
                  <OperationalTagPanel
                    actionData={actionData}
                    conditionData={conditionData}
                    recommendationMetrics={recommendationMetrics}
                  />
                ) : null}

                {/* Class × Severity cross-table — always shown when detections exist */}
                {chartDetections.length > 0 ? (
                  <div className="enterprise-panel border border-white/10 bg-[#111820] p-4">
                    <div className="mb-3 flex items-center justify-between">
                      <div>
                        <h2 className="text-sm font-semibold text-white">What Was Found, and How Bad?</h2>
                        <p className="mt-1 text-xs text-slate-400">
                          For each type of defect, how many were minor, moderate, critical, or unclear.
                        </p>
                      </div>
                      <ShieldAlert className="h-4 w-4 text-slate-400" />
                    </div>
                    <ClassSeverityTable rows={classSevData} />
                  </div>
                ) : null}

                {analyticsMode && chartDetections.length > 0 ? (
                  <>
                    <div className="enterprise-panel border border-white/10 bg-[#111820] p-4">
                      <div className="mb-3 flex items-center justify-between">
                        <div>
                          <h2 className="text-sm font-semibold text-white">How Sure Was the AI?</h2>
                          <p className="mt-1 text-xs text-slate-400">
                            Overall confidence per defect type. Low-confidence detections may need a manual second look.
                          </p>
                        </div>
                        <Gauge className="h-4 w-4 text-slate-400" />
                      </div>
                      <ClassConfidencePanel rows={classConfData} />
                    </div>

                    <div className="enterprise-panel border border-white/10 bg-[#111820] p-4">
                      <div className="mb-3 flex items-center justify-between">
                        <div>
                          <h2 className="text-sm font-semibold text-white">How Large Are the Defects?</h2>
                          <p className="mt-1 text-xs text-slate-400">
                            Average portion of the image each defect type covers. Larger area often means more visible damage.
                          </p>
                        </div>
                        <ZoomIn className="h-4 w-4 text-slate-400" />
                      </div>
                      <div className="h-[220px]">
                        <AreaCoverageChart rows={areaData} />
                      </div>
                    </div>
                  </>
                ) : null}
              </section>

              <section
                className={analyticsMode ? "grid min-h-0 gap-5 lg:grid-cols-2" : "flex min-h-0 flex-col gap-5"}
              >
                {!analyticsMode ? (
                  <div className="enterprise-panel border border-white/10 bg-[#111820] p-4">
                    <div className="mb-3 flex items-center justify-between">
                      <h2 className="text-sm font-semibold text-white">Case Controls</h2>
                      <Camera className="h-4 w-4 text-slate-400" />
                    </div>
                    <p className="text-sm leading-6 text-slate-400">
                      Start a new inspection when the case queue is empty, or reopen saved evidence from History for focused review.
                    </p>
                    <div className="mt-4 grid gap-2 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2">
                      <Link
                        href="/monitoring"
                        className="theme-primary-button inline-flex h-11 items-center justify-center gap-2 rounded-lg px-4 text-sm font-semibold text-slate-950"
                      >
                        <Camera className="h-4 w-4" />
                        New inspection
                      </Link>
                      <Link
                        href="/history"
                        className="theme-ghost-button inline-flex h-11 items-center justify-center gap-2 rounded-lg border border-white/10 bg-white/[0.04] px-4 text-sm font-semibold text-slate-100"
                      >
                        <History className="h-4 w-4" />
                        Open history
                      </Link>
                    </div>
                  </div>
                ) : null}
                <StatusPanel apiOnline={apiOnline} apiMessage={apiMessage} busy={busy} scanning={scanning} />
                <PipelinePanel
                  stage1Latency={latest.stageMetrics.stage1LatencyMs}
                  stage2Latency={latest.stageMetrics.stage2LatencyMs}
                  cropsClassified={latest.stageMetrics.cropsClassified}
                  detectorModel={
                    latest.stageMetrics.detectorModel === "not selected"
                      ? modelDisplayName(settings.detectorPath)
                      : latest.stageMetrics.detectorModel
                  }
                  severityModel={
                    latest.stageMetrics.severityModel === "not selected"
                      ? modelDisplayName(settings.severityPath)
                      : latest.stageMetrics.severityModel
                  }
                />
                {!analyticsMode ? (
                  <div ref={historyRef}>
                    <HistoryPanel
                      history={history}
                      expandedHistoryId={expandedHistoryId}
                      onToggleExpanded={(id) => setExpandedHistoryId((current) => (current === id ? "" : id))}
                      onRestoreRecord={restoreHistoryRecord}
                      onRestoreFrame={restoreVideoFrame}
                    />
                  </div>
                ) : null}
              </section>
            </div>
          )) : (
          <div
            ref={monitorRef}
            className={`grid p-4 xl:grid-cols-[minmax(220px,0.72fr)_minmax(0,1.62fr)_minmax(260px,0.86fr)] xl:gap-4 2xl:grid-cols-[minmax(260px,0.75fr)_minmax(0,1.8fr)_minmax(300px,0.9fr)] 2xl:gap-5 2xl:p-5 ${
              monitorFixed ? "h-full min-h-0 overflow-hidden" : "min-h-[calc(100vh-86px)] items-start overflow-visible"
            }`}
          >
            <section
              className={`order-2 grid min-w-0 gap-3 md:grid-cols-2 xl:order-1 xl:grid-cols-1 ${
                monitorFixed ? "h-full min-h-0 overflow-hidden" : "h-auto overflow-visible"
              }`}
            >
              <StatusPanel apiOnline={apiOnline} apiMessage={apiMessage} busy={busy} scanning={scanning} compact />
              <PipelinePanel
                stage1Latency={latest.stageMetrics.stage1LatencyMs}
                stage2Latency={latest.stageMetrics.stage2LatencyMs}
                cropsClassified={latest.stageMetrics.cropsClassified}
                detectorModel={
                  latest.stageMetrics.detectorModel === "not selected"
                    ? modelDisplayName(settings.detectorPath)
                    : latest.stageMetrics.detectorModel
                }
                severityModel={
                  latest.stageMetrics.severityModel === "not selected"
                    ? modelDisplayName(settings.severityPath)
                    : latest.stageMetrics.severityModel
                  }
                compact
              />
            </section>

            <section
              className={`order-1 flex min-w-0 flex-col xl:order-2 ${
                monitorFixed ? "h-full min-h-0" : "min-h-[calc(100vh-108px)]"
              }`}
            >
              <div
                className={`enterprise-panel visual-console flex flex-1 flex-col border border-white/10 bg-[#12171d] shadow-2xl shadow-black/30 ${
                  monitorFixed ? "h-full min-h-0 overflow-hidden" : "min-h-[calc(100vh-108px)] overflow-visible"
                }`}
              >
                <div className="flex flex-col gap-3 border-b border-white/10 p-4 lg:flex-row lg:items-center lg:justify-between">
                  <div>
                    <h2 className="text-lg font-semibold text-white">Visual Evidence</h2>
                    <p className="text-sm text-slate-400">Camera, uploaded images, and annotated detections stay in the main workspace.</p>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <button
                      onClick={toggleLiveMonitoring}
                      className="theme-live-button inline-flex h-10 items-center gap-2 rounded-lg border border-teal-300/30 bg-teal-300/10 px-3 text-sm font-semibold text-teal-100 hover:bg-teal-300/15"
                    >
                      {scanning || (sourceType !== "video" && cameraActive) ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                      {sourceType === "video"
                        ? scanning
                          ? "Pause video scan"
                          : "Scan video"
                        : scanning || cameraActive
                          ? "Stop live monitor"
                          : "Live monitor"}
                    </button>
                    <button
                      onClick={runCurrentFrameOnce}
                      disabled={busy || (sourceType !== "video" && !cameraActive)}
                      className="inline-flex h-10 items-center gap-2 rounded-lg border border-amber-300/30 bg-amber-300/10 px-3 text-sm font-semibold text-amber-100 hover:bg-amber-300/15 disabled:cursor-not-allowed disabled:opacity-45"
                    >
                      <CircleDot className="h-4 w-4" />
                      Analyze frame
                    </button>
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="theme-ghost-button inline-flex h-10 items-center gap-2 rounded-lg border border-white/10 bg-white/[0.04] px-3 text-sm font-semibold text-slate-100 hover:bg-white/[0.08]"
                    >
                      <Upload className="h-4 w-4" />
                      Upload image/video
                    </button>
                    <button
                      onClick={() => setShowLinkInput((value) => !value)}
                      className="theme-link-button inline-flex h-10 items-center gap-2 rounded-lg border border-violet-300/30 bg-violet-300/10 px-3 text-sm font-semibold text-violet-100 hover:bg-violet-300/15"
                    >
                      <Link2 className="h-4 w-4" />
                      Link
                    </button>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/*,video/*"
                      className="hidden"
                      onChange={(event) => onUploadMedia(event.target.files?.[0])}
                    />
                  </div>
                </div>
                {SAMPLE_IMAGES.length > 0 ? (
                  <div className="border-b border-white/10 bg-[#0b1016] px-4 py-3">
                    <p className="mb-2 text-xs font-semibold uppercase tracking-[0.13em] text-slate-500">Try a sample image</p>
                    <div className="flex flex-wrap gap-2">
                      {SAMPLE_IMAGES.map((sample) => (
                        <button
                          key={sample.filename}
                          type="button"
                          className="group flex flex-col items-start rounded-lg border border-white/[0.08] bg-white/[0.03] px-3 py-2 text-left transition hover:border-teal-300/30 hover:bg-teal-300/[0.05] active:scale-[0.97]"
                          onClick={async () => {
                            try {
                              const res = await fetch(`/samples/${sample.filename}`);
                              if (!res.ok) return;
                              const blob = await res.blob();
                              const file = new File([blob], sample.filename, { type: blob.type || "image/jpeg" });
                              onUploadMedia(file);
                            } catch {
                              // silently ignore network errors
                            }
                          }}
                        >
                          <span className="text-xs font-semibold text-slate-200 group-hover:text-teal-100">{sample.label}</span>
                          <span className="mt-0.5 text-[10px] text-slate-500 group-hover:text-teal-300/70">{sample.defectHint}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                ) : null}
                {showLinkInput ? (
                  <div className="grid gap-3 border-b border-white/10 bg-[#0b1016] p-4 2xl:grid-cols-[minmax(0,1fr)_130px_120px_130px_auto]">
                    <input
                      className="h-10 rounded-lg border border-white/10 bg-[#111820] px-3 text-sm text-slate-100 outline-none placeholder:text-slate-500 focus:border-violet-300/60"
                      value={mediaUrl}
                      onChange={(event) => setMediaUrl(event.target.value)}
                      placeholder="Public Drive file URL, image/video URL, or YouTube link"
                    />
                    <label className="grid grid-cols-[1fr_64px] items-center gap-2 text-xs uppercase tracking-[0.16em] text-slate-500">
                      Start
                      <input
                        className="h-10 rounded-lg border border-white/10 bg-[#111820] px-2 text-sm text-slate-100 outline-none focus:border-violet-300/60"
                        type="number"
                        min={0}
                        step={1}
                        value={linkFrameTime}
                        onChange={(event) => setLinkFrameTime(Math.max(0, Number(event.target.value) || 0))}
                      />
                    </label>
                    <label className="grid grid-cols-[1fr_58px] items-center gap-2 text-xs uppercase tracking-[0.16em] text-slate-500">
                      FPS
                      <input
                        className="h-10 rounded-lg border border-white/10 bg-[#111820] px-2 text-sm text-slate-100 outline-none focus:border-violet-300/60"
                        type="number"
                        min={0.1}
                        max={maxLinkedVideoFps}
                        step={0.1}
                        value={linkScanFps}
                        onChange={(event) => setLinkScanFps(Math.max(0.1, Math.min(maxLinkedVideoFps, Number(event.target.value) || 1)))}
                      />
                    </label>
                    <label className="inline-flex h-10 items-center justify-center gap-2 rounded-lg border border-white/10 bg-[#111820] px-3 text-xs font-semibold uppercase tracking-[0.12em] text-slate-300">
                      <input
                        type="checkbox"
                        checked={scanLinkedVideo}
                        onChange={(event) => setScanLinkedVideo(event.target.checked)}
                        className="h-4 w-4 accent-violet-300"
                      />
                      Scan
                    </label>
                    <button
                      onClick={inferLinkedMedia}
                      disabled={busy || Boolean(activeVideoJob)}
                      className="theme-secondary-button inline-flex h-10 items-center justify-center gap-2 rounded-lg bg-violet-300 px-4 text-sm font-semibold text-slate-950 transition hover:bg-violet-200 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      <Link2 className="h-4 w-4" />
                      {activeVideoJob ? "Task running" : "Analyze media"}
                    </button>
                  </div>
                ) : null}
                <div className="max-h-10 overflow-hidden border-b border-white/10 bg-white/[0.02] px-4 py-2 text-[11px] leading-4 text-slate-400">
                  <strong className="text-slate-200">Live monitor</strong> samples camera frames.{" "}
                  <strong className="text-slate-200">Analyze frame</strong> runs one visible frame.{" "}
                  <strong className="text-slate-200">Link</strong> accepts Drive, direct media, or YouTube and returns clickable defect frames.
                </div>
                <div className="grid grid-cols-4 gap-2 border-b border-white/10 bg-white/[0.025] p-2">
                  <TelemetryChip icon={Activity} label="live throughput" value={estimatedThroughput} tone="text-cyan-200" />
                  <TelemetryChip icon={RefreshCcw} label="capture cadence" value={frameCadence} tone="text-teal-200" />
                  <TelemetryChip icon={ImageUp} label="active findings" value={String(latest.detections.length)} tone="text-amber-200" />
                  <TelemetryChip icon={Video} label="video queue" value={queueStatus} tone="text-violet-200" />
                </div>

                <div
                  className={`relative w-full bg-black ${
                    monitorReviewMode
                      ? "h-[calc(100vh-270px)] min-h-[520px] flex-1 overflow-auto"
                      : "min-h-[180px] flex-1 overflow-auto"
                  }`}
                >
                  <div
                    className="relative grid min-h-full min-w-full place-items-center"
                    style={{
                      width: `${Math.max(evidenceZoom, 1) * 100}%`,
                      height: `${Math.max(evidenceZoom, 1) * 100}%`,
                    }}
                  >
                    {showVideo ? (
                      <video
                        ref={videoRef}
                        src={sourceType === "video" ? videoObjectUrl : undefined}
                        className="h-full w-full object-contain"
                        autoPlay={sourceType === "camera"}
                        controls={sourceType === "video"}
                        muted
                        playsInline
                        onLoadedMetadata={() => {
                          if (sourceType === "video" && videoRef.current) {
                            videoRef.current.currentTime = Math.max(0, linkFrameTime);
                          }
                        }}
                      />
                    ) : evidenceImage ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={evidenceImage} alt="Inspection evidence" className="h-full w-full object-contain" />
                    ) : (
                      <div className="grid h-full min-h-[320px] w-full place-items-center bg-[radial-gradient(circle_at_center,rgba(45,212,191,0.14),transparent_45%),#05070a]">
                        <div className="max-w-sm text-center">
                          <div className="mx-auto mb-4 grid h-14 w-14 place-items-center rounded-lg border border-cyan-300/30 bg-cyan-300/10">
                            <Video className="h-7 w-7 text-cyan-100" />
                          </div>
                          <h3 className="text-lg font-semibold text-white">
                            {cameraError ? "Camera is not available" : "Start live monitor or upload evidence"}
                          </h3>
                          <p className="mt-2 text-sm text-slate-400">
                            {cameraError || cameraMessage || "The annotated feed will stay here while the palettes update around it."}
                          </p>
                        </div>
                      </div>
                    )}

                    {showInteractiveDetections ? latest.detections.map((detection) => (
                      <button
                        key={detection.id}
                        className={`absolute border-2 text-left shadow-[0_0_24px_rgba(0,0,0,0.55)] transition ${
                          detection.id === selectedDetection?.id ? "border-white" : "border-cyan-300"
                        }`}
                        style={{
                          left: pct(detection.bbox.x),
                          top: pct(detection.bbox.y),
                          width: pct(detection.bbox.width),
                          height: pct(detection.bbox.height),
                        }}
                        onClick={() => selectDetection(detection.id)}
                      >
                        <span
                          className="absolute -top-7 left-0 rounded-md px-2 py-1 text-xs font-semibold text-slate-950"
                          style={{ background: severityColors[detection.severity] }}
                        >
                          {detection.className} | {detection.severity}
                        </span>
                      </button>
                    )) : null}
                  </div>

                  {showVideo || evidenceImage || latest.detections.length ? (
                    <div className="absolute right-3 top-3 z-30 flex items-center gap-1 rounded-lg border border-white/10 bg-[#05070a]/90 p-1 shadow-2xl backdrop-blur">
                      <button
                        type="button"
                        onClick={() => setEvidenceZoom((value) => Math.max(1, Number((value - 0.25).toFixed(2))))}
                        className="grid h-8 w-8 place-items-center rounded-md text-slate-200 transition hover:bg-white/10"
                        title="Zoom out"
                        aria-label="Zoom out evidence"
                      >
                        <ZoomOut className="h-4 w-4" />
                      </button>
                      <span className="w-12 text-center font-mono text-xs font-semibold text-white">{zoomPercent}%</span>
                      <button
                        type="button"
                        onClick={() => setEvidenceZoom((value) => Math.min(3, Number((value + 0.25).toFixed(2))))}
                        className="grid h-8 w-8 place-items-center rounded-md text-slate-200 transition hover:bg-white/10"
                        title="Zoom in"
                        aria-label="Zoom in evidence"
                      >
                        <ZoomIn className="h-4 w-4" />
                      </button>
                      <button
                        type="button"
                        onClick={() => setEvidenceZoom(1)}
                        className="grid h-8 w-8 place-items-center rounded-md text-slate-200 transition hover:bg-white/10"
                        title="Reset zoom"
                        aria-label="Reset evidence zoom"
                      >
                        <RefreshCcw className="h-4 w-4" />
                      </button>
                    </div>
                  ) : null}

                  {!showVideo && latest.source?.sampleCount ? (
                    <div className="absolute left-4 top-4 max-w-[calc(100%-2rem)] rounded-lg border border-violet-300/30 bg-[#05070a]/85 px-3 py-2 text-xs text-violet-50 shadow-xl backdrop-blur">
                      Best frame {latest.source.selectedFrameTimeSec?.toFixed(1) ?? "0.0"}s from {latest.source.sampleCount} sampled frame
                      {latest.source.sampleCount === 1 ? "" : "s"} @ {latest.source.scanFps ?? 1} FPS
                      {latest.source.safetyCapped ? " | capped for runtime safety" : ""}
                      {latest.source.reviewMode ? ` | review sensitivity ${latest.source.usedConfidence}` : ""}
                    </div>
                  ) : null}

                  {analysisActive ? (
                    <div className="absolute left-3 top-3 z-20 max-w-[min(360px,calc(100%-1.5rem))] rounded-lg border border-violet-300/30 bg-[#0b1016]/95 p-3 text-left shadow-2xl backdrop-blur">
                      <div className="flex items-center gap-2">
                        <div className="h-4 w-4 animate-spin rounded-full border-2 border-violet-200 border-t-transparent" />
                        <p className="text-xs font-semibold text-white">Analyzing evidence</p>
                      </div>
                      <p className="mt-2 text-[11px] leading-4 text-slate-300">
                        Sampling at {linkScanFps} FPS, detecting defects, cropping ROIs, then classifying severity.
                      </p>
                      <div className="mt-2 grid grid-cols-3 gap-1 text-[10px]">
                        <span className="rounded-md border border-white/10 bg-white/[0.04] px-2 py-1.5 text-slate-300">
                          elapsed<br />
                          <strong className="text-white">{analysisElapsedSec}s</strong>
                        </span>
                        <span className="rounded-md border border-white/10 bg-white/[0.04] px-2 py-1.5 text-slate-300">
                          FPS<br />
                          <strong className="text-white">{linkScanFps}</strong>
                        </span>
                        <span className="rounded-md border border-white/10 bg-white/[0.04] px-2 py-1.5 text-slate-300">
                          mode<br />
                          <strong className="text-white">{scanLinkedVideo ? "scan" : "frame"}</strong>
                        </span>
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
            </section>

            <section
              className={`order-3 min-w-0 xl:order-3 ${
                monitorFixed ? "h-full min-h-0" : "h-auto min-h-[520px]"
              }`}
            >
              <div
                className={`enterprise-panel flex flex-col border border-white/10 bg-[#111820] ${
                  monitorFixed ? "h-full min-h-0 overflow-hidden" : "min-h-[520px] overflow-visible"
                }`}
              >
                <div className="border-b border-white/10 p-4">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <h2 className="text-sm font-semibold text-white">Inspection Inspector</h2>
                      <p className="mt-1 text-xs text-slate-400">
                        {latest.detections.length} findings
                        {latest.source?.sampleCount ? ` | ${latest.source.sampleCount} video frames sampled` : ""}
                      </p>
                    </div>
                    <Layers className="h-4 w-4 text-slate-400" />
                  </div>
                  <div className="mt-4 grid grid-cols-2 gap-2 2xl:grid-cols-4">
                    {monitorPanels.map(({ id, label, icon: Icon }) => (
                      <button
                        key={id}
                        onClick={() => setMonitorPanel(id)}
                        className={`flex h-9 items-center justify-center gap-1 rounded-lg border px-1 text-[11px] font-semibold transition ${
                          monitorPanel === id
                            ? "border-cyan-300/50 bg-cyan-300/[0.12] text-cyan-100"
                            : "border-white/10 bg-white/[0.03] text-slate-300 hover:bg-white/[0.07]"
                        }`}
                      >
                        <Icon className="h-3.5 w-3.5" />
                        <span className="truncate">{label}</span>
                      </button>
                    ))}
                  </div>
                </div>

                <div className="min-h-0 flex-1 overflow-auto p-3">
                  {monitorPanel === "frames" ? (
                    latest.source?.sampleCount ? (
                      <div className="space-y-2">
                        <div className="rounded-lg border border-violet-300/20 bg-violet-300/[0.06] p-3">
                          <p className="text-sm font-semibold text-white">Video Defect Frames</p>
                          <p className="mt-1 text-xs text-slate-400">
                            {latest.source.positiveFrameCount ?? latestDefectFrames.length} positive frame
                            {(latest.source.positiveFrameCount ?? latestDefectFrames.length) === 1 ? "" : "s"} from{" "}
                            {latest.source.sampleCount} sampled
                          </p>
                        </div>
                        {latestDefectFrames.length ? (
                          latestDefectFrames.slice(0, 24).map((frame) => {
                            const selectedFrame =
                              latest.source?.selectedFrameTimeSec !== undefined &&
                              Math.abs(frame.frameTimeSec - latest.source.selectedFrameTimeSec) < 0.2;
                            return (
                              <button
                                key={frame.id}
                                onClick={() => restoreVideoFrame(frame)}
                                className={`grid w-full grid-cols-[72px_1fr_auto] items-center gap-2 rounded-lg border px-3 py-2 text-left text-xs transition ${
                                  selectedFrame
                                    ? "border-violet-300/50 bg-violet-300/[0.12]"
                                    : "border-white/10 bg-white/[0.03] hover:bg-white/[0.07]"
                                }`}
                              >
                                <span className="font-mono font-semibold text-violet-100">{frame.frameTimeSec.toFixed(1)}s</span>
                                <span className="min-w-0">
                                  <span className="block truncate text-slate-200">{frame.classes.join(", ")}</span>
                                  <span className="text-slate-500">
                                    {frame.detections} box{frame.detections === 1 ? "" : "es"} at threshold {frame.confidence}
                                  </span>
                                </span>
                                <span className="text-slate-300">{Math.round(frame.maxConfidence * 100)}%</span>
                              </button>
                            );
                          })
                        ) : (
                          <p className="rounded-lg border border-white/10 bg-white/[0.03] p-3 text-sm leading-6 text-slate-400">
                            This scan did not return defect-positive frames at the current confidence. Lower confidence or inspect a
                            different timestamp if the visual evidence clearly contains defects.
                          </p>
                        )}
                      </div>
                    ) : (
                      <EmptyMini label="Video frame results appear after a linked or uploaded video scan" />
                    )
                  ) : null}

                  {monitorPanel === "findings" ? (
                    latest.detections.length ? (
                      <div className="space-y-3">
                        {latest.detections.map((detection) => {
                          const actionMeta = priorityMeta[detection.priority];
                          const ActionIcon = actionMeta.icon;
                          return (
                            <button
                              key={detection.id}
                              onClick={() => selectDetection(detection.id)}
                              className={`w-full rounded-lg border p-3 text-left transition ${
                                selectedDetection?.id === detection.id
                                  ? "border-cyan-300/60 bg-cyan-300/10"
                                  : "border-white/10 bg-white/[0.03] hover:bg-white/[0.06]"
                              }`}
                            >
                              <div className="flex items-start justify-between gap-3">
                                <div>
                                  <p className="font-semibold text-white">{detection.className}</p>
                                  <p className="mt-1 text-xs text-slate-400">
                                    detector {Math.round(detection.confidence * 100)}% | severity{" "}
                                    {Math.round(detection.severityConfidence * 100)}%
                                  </p>
                                </div>
                                <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-1 text-xs ${priorityClasses[detection.priority]}`}>
                                  <ActionIcon className="h-3 w-3" />
                                  {actionMeta.label}
                                </span>
                              </div>
                              <div className="mt-3 flex items-center gap-2">
                                <span
                                  className="h-2.5 w-2.5 rounded-full"
                                  style={{ backgroundColor: severityColors[detection.severity] }}
                                />
                                <span className="text-xs capitalize text-slate-300">{detection.severity}</span>
                                <span className="text-xs text-slate-500">area {Math.round(detection.areaRatio * 1000) / 10}%</span>
                              </div>
                              <p className="mt-2 line-clamp-2 text-xs leading-5 text-slate-400">{actionMeta.description}</p>
                            </button>
                          );
                        })}
                      </div>
                    ) : latest.latencyMs > 0 ? (
                      <NoDetectionNotice
                        confidence={settings.confidence}
                        onReviewSensitivity={() => {
                          setMode("Multi-defect review");
                          setApiMessage("Multi-defect review enabled. Re-run inference to include separated lower-confidence candidates.");
                          setSettingsPanelOpen(true);
                        }}
                      />
                    ) : (
                      <EmptyMini label="No active findings" />
                    )
                  ) : null}

                  {monitorPanel === "selected" ? (
                    selectedDetection ? (
                      <div className="space-y-3">
                        <p className="text-lg font-semibold text-white">{selectedDetection.className}</p>
                        <div className="rounded-lg border border-cyan-300/20 bg-cyan-300/5 p-3">
                          <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] text-cyan-200">
                            <Camera className="h-4 w-4" />
                            Stage 1 Object Detection
                          </div>
                          <div className="grid grid-cols-2 gap-3 text-sm">
                            <Info label="Defect class" value={selectedDetection.className} />
                            <Info label="Detector conf" value={Math.round(selectedDetection.confidence * 100) + "%"} />
                            <Info label="Box area" value={`${Math.round(selectedDetection.areaRatio * 1000) / 10}%`} />
                            <Info label="Crop count" value={String(latest.stageMetrics.cropsClassified)} />
                          </div>
                        </div>
                        <div className="rounded-lg border border-teal-300/20 bg-teal-300/5 p-3">
                          <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] text-teal-200">
                            <Layers className="h-4 w-4" />
                            Stage 2 Severity Classification
                          </div>
                          <div className="grid grid-cols-2 gap-3 text-sm">
                            <Info label="Severity" value={selectedDetection.severity} />
                            <Info label="Severity conf" value={Math.round(selectedDetection.severityConfidence * 100) + "%"} />
                            <Info label="Priority" value={selectedDetection.priority} />
                            <Info label="Action tier" value={selectedDetection.priority} />
                          </div>
                        </div>
                        <ActionPlanCard detection={selectedDetection} />
                      </div>
                    ) : (
                      <p className="text-sm text-slate-400">Select a finding from the visual evidence or defect list.</p>
                    )
                  ) : null}

                  {monitorPanel === "charts" ? (
                    <div className="grid gap-4">
                      <div>
                        <div className="mb-3 flex items-center justify-between">
                          <h3 className="text-sm font-semibold text-white">Defect Distribution</h3>
                          <RefreshCcw className="h-4 w-4 text-slate-400" />
                        </div>
                        <div className="h-52">
                          {defectData.length ? <DefectBarChart data={defectData} /> : <EmptyMini label="Distribution appears after inference" />}
                        </div>
                      </div>
                      <div>
                        <div className="mb-3 flex items-center justify-between">
                          <h3 className="text-sm font-semibold text-white">Severity Mix</h3>
                          <Layers className="h-4 w-4 text-slate-400" />
                        </div>
                        <div className="h-52">
                          {severityData.length ? <SeverityPieChart data={severityData} /> : <EmptyMini label="No severity data yet" />}
                        </div>
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
            </section>

            <section className="hidden">
              <div className="enterprise-panel border border-white/10 bg-[#111820] p-4">
                <div className="mb-3 flex items-center justify-between">
                  <h2 className="text-sm font-semibold text-white">Condition Index Over Time</h2>
                  <Activity className="h-4 w-4 text-slate-400" />
                </div>
                <div className="h-56">
                  <ConditionTrendChart data={trendData} />
                </div>
              </div>

              <div className="enterprise-panel border border-white/10 bg-[#111820] p-4">
                <div className="mb-3 flex items-center justify-between">
                  <h2 className="text-sm font-semibold text-white">Inspection History</h2>
                  <History className="h-4 w-4 text-slate-400" />
                </div>
                <div className="max-h-56 overflow-auto">
                  {history.length ? (
                    <div className="space-y-2">
                      {history.slice(0, 8).map((record) => {
                        const defectFrames = record.frames?.filter((frame) => frame.detections > 0) ?? [];
                        const expanded = expandedHistoryId === record.id;
                        return (
                          <div key={record.id} className="rounded-lg border border-white/10 bg-white/[0.03]">
                            <div className="grid grid-cols-[1fr_auto_auto] gap-2 p-3">
                              <button
                                onClick={() => restoreHistoryRecord(record)}
                                className="min-w-0 text-left"
                              >
                                <span className="block truncate text-sm font-semibold text-white">{record.asset}</span>
                                <span className="text-xs text-slate-400">
                                  {record.createdAt} | {record.source}
                                  {record.source === "video"
                                    ? ` | ${record.scannedFrames ?? record.stageMetrics.sampledFrames ?? 1} scanned, ${defectFrames.length} defect frames`
                                    : ` | ${record.detections.length} defects`}
                                </span>
                              </button>
                              {record.source === "video" ? (
                                <button
                                  className="grid h-8 w-8 place-items-center rounded-md border border-white/10 bg-white/[0.04] text-slate-300"
                                  onClick={() => setExpandedHistoryId(expanded ? "" : record.id)}
                                  title={expanded ? "Hide defect frames" : "Show defect frames"}
                                >
                                  {expanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                                </button>
                              ) : (
                                <span />
                              )}
                              <span className={`text-sm font-semibold ${scoreTone(record.conditionIndex)}`}>
                                {record.conditionIndex.toFixed(0)}
                              </span>
                            </div>
                            {expanded ? (
                              <div className="border-t border-white/10 px-3 py-2">
                                {defectFrames.length ? (
                                  <div className="space-y-1.5">
                                    {defectFrames.slice(0, 12).map((frame) => (
                                      <button
                                        key={frame.id}
                                        onClick={() => restoreVideoFrame(frame, record)}
                                        className="grid w-full grid-cols-[64px_1fr_auto] items-center gap-2 rounded-md bg-white/[0.04] px-2 py-2 text-left text-xs hover:bg-white/[0.08]"
                                      >
                                        <span className="font-mono text-slate-200">{frame.frameTimeSec.toFixed(1)}s</span>
                                        <span className="truncate text-slate-300">{frame.classes.join(", ")}</span>
                                        <span className="text-slate-400">{Math.round(frame.maxConfidence * 100)}%</span>
                                      </button>
                                    ))}
                                  </div>
                                ) : (
                                  <p className="py-2 text-xs text-slate-400">No defect frames captured in this video inspection.</p>
                                )}
                              </div>
                            ) : null}
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <EmptyMini label="Inspection records save here" />
                  )}
                </div>
              </div>
            </section>
          </div>
          )}
        </section>
      </div>
      {dragActive ? (
        <div className="pointer-events-none fixed inset-0 z-50 grid place-items-center bg-cyan-950/45 p-6 backdrop-blur-sm">
          <div className="w-full max-w-lg rounded-xl border border-cyan-300/50 bg-[#071116]/95 p-8 text-center shadow-2xl shadow-black/40">
            <div className="mx-auto mb-4 grid h-14 w-14 place-items-center rounded-lg border border-cyan-300/40 bg-cyan-300/10">
              <Upload className="h-7 w-7 text-cyan-100" />
            </div>
            <h2 className="text-xl font-semibold text-white">Drop evidence to analyze</h2>
            <p className="mt-2 text-sm leading-6 text-slate-300">
              Images run inference immediately. Videos load into the monitoring workspace for frame analysis or scanning.
            </p>
          </div>
        </div>
      ) : null}
      {workspaceView === "monitor" && videoJob ? (
        <div className="fixed right-4 top-24 z-40 w-[min(430px,calc(100vw-2rem))]">
          <VideoJobPanel
            job={videoJob}
            onCancel={cancelVideoJob}
            onLoadResult={() => loadVideoJobResult(videoJob)}
          />
        </div>
      ) : null}
      <canvas ref={canvasRef} className="hidden" />
    </main>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  suffix,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  suffix?: string;
  onChange: (value: number) => void;
}) {
  return (
    <label className="text-sm text-slate-300">
      <span className="mb-2 flex items-center justify-between text-xs uppercase tracking-[0.16em] text-slate-500">
        {label}
        <strong className="font-mono text-slate-200">
          {suffix ? `${value}${suffix}` : value.toFixed(2)}
        </strong>
      </span>
      <input
        className="h-2 w-full accent-teal-300"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function StatusPanel({
  apiOnline,
  apiMessage,
  busy,
  scanning,
  compact = false,
}: {
  apiOnline: boolean;
  apiMessage: string;
  busy: boolean;
  scanning: boolean;
  compact?: boolean;
}) {
  return (
    <div className={`enterprise-panel border border-white/10 bg-[#111820] ${compact ? "p-3" : "p-4"}`}>
      <div className={`${compact ? "mb-2" : "mb-4"} flex items-center justify-between`}>
        <h2 className="text-sm font-semibold text-white">System Status</h2>
        <Server className="h-4 w-4 text-slate-400" />
      </div>
      <div className={compact ? "space-y-2" : "space-y-3"}>
        <StatusRow active={apiOnline} label="Inference API" detail={apiMessage} compact={compact} />
        <StatusRow active={scanning} label="Live monitor" detail={scanning ? "Frame sampling enabled" : "Manual capture"} compact={compact} />
        <StatusRow active={!busy} label="Processing" detail={busy ? "Inference running" : "Ready"} compact={compact} />
      </div>
    </div>
  );
}

function StatusRow({ active, label, detail, compact = false }: { active: boolean; label: string; detail: string; compact?: boolean }) {
  return (
    <div className={`flex gap-2 rounded-lg border border-white/10 bg-white/[0.03] ${compact ? "p-2" : "p-3"}`}>
      {active ? (
        <CheckCircle2 className={`${compact ? "h-3.5 w-3.5" : "h-4 w-4"} mt-0.5 text-teal-300`} />
      ) : (
        <AlertTriangle className={`${compact ? "h-3.5 w-3.5" : "h-4 w-4"} mt-0.5 text-amber-300`} />
      )}
      <div className="min-w-0">
        <p className={`${compact ? "text-xs" : "text-sm"} font-semibold text-white`}>{label}</p>
        <p className="truncate text-[11px] text-slate-400">{detail}</p>
      </div>
    </div>
  );
}

function PipelinePanel({
  stage1Latency,
  stage2Latency,
  cropsClassified,
  detectorModel,
  severityModel,
  compact = false,
}: {
  stage1Latency: number;
  stage2Latency: number;
  cropsClassified: number;
  detectorModel: string;
  severityModel: string;
  compact?: boolean;
}) {
  return (
    <div className={`enterprise-panel border border-white/10 bg-[#111820] ${compact ? "p-3" : "p-4"}`}>
      <div className={`${compact ? "mb-2" : "mb-4"} flex items-center justify-between`}>
        <h2 className="text-sm font-semibold text-white">Two-Stage Inference</h2>
        <Zap className="h-4 w-4 text-teal-200" />
      </div>
      <div className={compact ? "space-y-2" : "space-y-3"}>
        <StageRow
          accent="cyan"
          icon={Camera}
          title="Stage 1 Detector"
          metric={formatMs(stage1Latency)}
          detail={`${detectorModel} | boxes and defect classes`}
          compact={compact}
        />
        <div className={`${compact ? "h-2" : "h-5"} ml-5 border-l border-dashed border-white/20`} />
        <StageRow
          accent="teal"
          icon={Layers}
          title="Stage 2 Severity"
          metric={formatMs(stage2Latency)}
          detail={`${severityModel} | ${cropsClassified} cropped regions classified`}
          compact={compact}
        />
      </div>
    </div>
  );
}

function StageRow({
  icon: Icon,
  title,
  metric,
  detail,
  accent,
  compact = false,
}: {
  icon: LucideIcon;
  title: string;
  metric: string;
  detail: string;
  accent: "cyan" | "teal";
  compact?: boolean;
}) {
  const color = accent === "cyan" ? "text-cyan-200 bg-cyan-300/10 border-cyan-300/25" : "text-teal-200 bg-teal-300/10 border-teal-300/25";
  return (
    <div className={`rounded-lg border border-white/10 bg-white/[0.03] ${compact ? "p-2" : "p-3"}`}>
      <div className="flex items-start justify-between gap-3">
        <div className={`flex min-w-0 ${compact ? "gap-2" : "gap-3"}`}>
          <div className={`grid ${compact ? "h-8 w-8" : "h-9 w-9"} shrink-0 place-items-center rounded-lg border ${color}`}>
            <Icon className={compact ? "h-3.5 w-3.5" : "h-4 w-4"} />
          </div>
          <div className="min-w-0">
            <p className={`${compact ? "text-xs" : "text-sm"} font-semibold text-white`}>{title}</p>
            <p className="truncate text-[11px] text-slate-400">{detail}</p>
          </div>
        </div>
        <span className="shrink-0 rounded-md border border-white/10 bg-black/20 px-2 py-1 font-mono text-xs text-slate-200">
          {metric}
        </span>
      </div>
    </div>
  );
}

function MetricPalette({
  detections,
  critical,
  immediate,
  latency,
  condition,
}: {
  detections: number;
  critical: number;
  immediate: number;
  latency: number;
  condition: number;
}) {
  const items = [
    { label: "Detected defects", value: detections, color: "text-cyan-200", icon: ImageUp, progress: Math.min(100, detections * 14) },
    { label: "Critical severity", value: critical, color: "text-rose-200", icon: ShieldAlert, progress: Math.min(100, critical * 28) },
    { label: "Immediate work", value: immediate, color: "text-amber-200", icon: AlertTriangle, progress: Math.min(100, immediate * 28) },
    { label: "Condition index", value: condition.toFixed(1), color: scoreTone(condition), icon: Gauge, progress: condition },
    { label: "Last latency", value: formatMs(latency), color: "text-violet-200", icon: Cpu, progress: Math.min(100, latency / 20) },
  ];

  return (
    <div className="enterprise-panel border border-white/10 bg-[#111820] p-4">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-white">Inspection Analytics Summary</h2>
          <p className="mt-1 text-xs text-slate-400">Historical defect, severity, condition, and model-latency indicators.</p>
        </div>
        <Gauge className="h-4 w-4 text-slate-400" />
      </div>
      <div className="grid gap-3 md:grid-cols-2 2xl:grid-cols-5">
        {items.map((item) => (
          <div key={item.label} className="rounded-lg border border-white/10 bg-white/[0.03] p-3">
            <div className="flex items-center justify-between gap-3">
              <div className="min-w-0">
                <p className="truncate text-[11px] uppercase tracking-[0.14em] text-slate-500">{item.label}</p>
                <p className={`mt-1 text-lg font-semibold ${item.color}`}>{item.value}</p>
              </div>
              <item.icon className="h-4 w-4 shrink-0 text-slate-500" />
            </div>
            <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-black/25">
              <div
                className={`h-full rounded-full bg-current transition-all ${item.color}`}
                style={{ width: `${Math.max(4, Math.min(100, item.progress))}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function DashboardOpsPalette({
  history,
  apiOnline,
  busy,
  scanning,
  activeVideoJob,
  modelCount,
  estimatedThroughput,
  frameCadence,
  queueStatus,
}: {
  history: InspectionRecord[];
  apiOnline: boolean;
  busy: boolean;
  scanning: boolean;
  activeVideoJob: boolean;
  modelCount: number;
  estimatedThroughput: string;
  frameCadence: string;
  queueStatus: string;
}) {
  const positiveCases = history.filter((record) => detectionsForRecord(record).length > 0).length;
  const storedFrames = history.reduce((sum, record) => sum + (record.frames?.length ?? 0), 0);
  const items = [
    {
      label: "API readiness",
      value: apiOnline ? "Online" : "Offline",
      color: apiOnline ? "text-teal-200" : "text-amber-200",
      icon: Server,
      progress: apiOnline ? 100 : 10,
    },
    {
      label: "Model slots",
      value: `${Math.min(modelCount, 2)}/2`,
      color: modelCount >= 2 ? "text-cyan-200" : "text-amber-200",
      icon: Cpu,
      progress: Math.min(100, (modelCount / 2) * 100),
    },
    {
      label: "Saved inspections",
      value: history.length,
      color: "text-violet-200",
      icon: History,
      progress: Math.min(100, history.length * 12),
    },
    {
      label: "Review cases",
      value: positiveCases,
      color: "text-amber-200",
      icon: ImageUp,
      progress: Math.min(100, positiveCases * 18),
    },
    {
      label: "Stored frames",
      value: storedFrames,
      color: "text-sky-200",
      icon: Video,
      progress: Math.min(100, storedFrames / 4),
    },
    {
      label: "Processing state",
      value: activeVideoJob ? queueStatus : busy ? "Running" : scanning ? "Live" : "Idle",
      color: activeVideoJob || busy || scanning ? "text-teal-200" : "text-slate-200",
      icon: RefreshCcw,
      progress: activeVideoJob || busy || scanning ? 80 : 18,
    },
  ];

  return (
    <div className="enterprise-panel border border-white/10 bg-[#111820] p-4">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-white">Operations Readiness</h2>
          <p className="mt-1 text-xs text-slate-400">System, intake, and processing KPIs. Detailed defect analytics live on the Analytics page.</p>
        </div>
        <Server className="h-4 w-4 text-slate-400" />
      </div>
      <div className="grid gap-3 md:grid-cols-2 2xl:grid-cols-6">
        {items.map((item) => (
          <div key={item.label} className="rounded-lg border border-white/10 bg-white/[0.03] p-3">
            <div className="flex items-center justify-between gap-3">
              <div className="min-w-0">
                <p className="truncate text-[11px] uppercase tracking-[0.14em] text-slate-500">{item.label}</p>
                <p className={`mt-1 truncate text-lg font-semibold ${item.color}`}>{item.value}</p>
              </div>
              <item.icon className="h-4 w-4 shrink-0 text-slate-500" />
            </div>
            <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-black/25">
              <div
                className={`h-full rounded-full bg-current transition-all ${item.color}`}
                style={{ width: `${Math.max(4, Math.min(100, item.progress))}%` }}
              />
            </div>
          </div>
        ))}
      </div>
      <div className="mt-3 grid gap-2 md:grid-cols-2">
        <TelemetryChip icon={Activity} label="Runtime capacity" value={estimatedThroughput} tone="text-cyan-200" />
        <TelemetryChip icon={RefreshCcw} label="Capture cadence" value={frameCadence} tone="text-teal-200" />
      </div>
    </div>
  );
}

function DashboardOperationsOverview({
  history,
  videoJob,
  apiOnline,
  apiMessage,
  latest,
  estimatedThroughput,
  frameCadence,
  queueStatus,
  onRestoreRecord,
  onRestoreFrame,
}: {
  history: InspectionRecord[];
  videoJob: VideoJob | null;
  apiOnline: boolean;
  apiMessage: string;
  latest: InferenceResult;
  estimatedThroughput: string;
  frameCadence: string;
  queueStatus: string;
  onRestoreRecord: (record: InspectionRecord) => void;
  onRestoreFrame: (frame: VideoFrameSummary, record: InspectionRecord) => void;
}) {
  const positiveCases = history.filter((record) => detectionsForRecord(record).length > 0).length;
  const videoRecords = history.filter((record) => record.source === "video").length;
  const storedFrames = history.reduce((sum, record) => sum + (record.frames?.length ?? 0), 0);
  const sourceData = [
    { name: "Images", value: history.filter((record) => record.source === "image").length, color: "#38bdf8" },
    { name: "Videos", value: videoRecords, color: "#a78bfa" },
    { name: "Camera", value: history.filter((record) => record.source === "camera").length, color: "#2dd4bf" },
  ].filter((item) => item.value > 0);
  const coverageData = [
    { name: "Records", value: history.length, color: "#38bdf8" },
    { name: "Review cases", value: positiveCases, color: "#f59e0b" },
    { name: "Video records", value: videoRecords, color: "#a78bfa" },
    { name: "Stored frames", value: storedFrames, color: "#2dd4bf" },
  ].filter((item) => item.value > 0);
  const recentItems = history.slice(0, 3).map((record) => {
    const defectFrames = record.frames?.filter((frame) => frame.detections > 0) ?? [];
    const firstFrame = defectFrames.find((frame) => frame.evidenceImage || frame.annotatedImage);
    return {
      record,
      defectFrames,
      firstFrame,
      preview: firstFrame?.evidenceImage ?? firstFrame?.annotatedImage ?? previewForRecord(record),
      detectionCount: detectionsForRecord(record).length,
    };
  });
  const queueProgress = videoJob ? Math.max(0, Math.min(100, Math.round(videoJob.progress || 0))) : 0;

  return (
    <div className="grid gap-5">
      <div className="grid gap-5 2xl:grid-cols-[minmax(0,1fr)_minmax(330px,0.75fr)]">
        <div className="enterprise-panel border border-white/10 bg-[#111820] p-4">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <h2 className="text-sm font-semibold text-white">Inspection Intake Mix</h2>
              <p className="mt-1 text-xs text-slate-400">Historical saved records grouped by media source.</p>
            </div>
            <ImageUp className="h-4 w-4 text-slate-400" />
          </div>
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_210px]">
            <div className="h-52 rounded-lg border border-white/10 bg-black/10 p-2">
              {sourceData.length ? <MetricBarChart data={sourceData} /> : <EmptyMini label="Intake mix appears after saved inspections" />}
            </div>
            <div className="grid gap-2">
              <TelemetryChip icon={History} label="records" value={String(history.length)} tone="text-cyan-200" />
              <TelemetryChip icon={ImageUp} label="review cases" value={String(positiveCases)} tone="text-amber-200" />
              <TelemetryChip icon={Video} label="stored frames" value={String(storedFrames)} tone="text-violet-200" />
            </div>
          </div>
        </div>

        <div className="enterprise-panel border border-white/10 bg-[#111820] p-4">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <h2 className="text-sm font-semibold text-white">Processing Queue</h2>
              <p className="mt-1 text-xs text-slate-400">Current backend and video task state, not defect-condition scoring.</p>
            </div>
            <RefreshCcw className="h-4 w-4 text-slate-400" />
          </div>
          <div className="rounded-lg border border-white/10 bg-white/[0.03] p-3">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="truncate text-[11px] uppercase tracking-[0.14em] text-slate-500">Queue status</p>
                <p className="mt-1 truncate text-lg font-semibold text-white">{queueStatus}</p>
                <p className="mt-1 truncate text-xs text-slate-400">{videoJob?.message || apiMessage}</p>
              </div>
              {apiOnline ? (
                <CheckCircle2 className="h-4 w-4 shrink-0 text-teal-300" />
              ) : (
                <AlertTriangle className="h-4 w-4 shrink-0 text-amber-300" />
              )}
            </div>
            <div className="mt-3 h-2 overflow-hidden rounded-full bg-black/30">
              <div className="h-full rounded-full bg-violet-300 transition-all" style={{ width: `${videoJob ? queueProgress : apiOnline ? 100 : 10}%` }} />
            </div>
          </div>
          <div className="mt-3 grid gap-2 sm:grid-cols-2">
            <TelemetryChip icon={Activity} label="runtime" value={estimatedThroughput} tone="text-cyan-200" />
            <TelemetryChip icon={RefreshCcw} label="cadence" value={frameCadence} tone="text-teal-200" />
          </div>
        </div>
      </div>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,0.95fr)_minmax(360px,1.05fr)]">
        <div className="enterprise-panel border border-white/10 bg-[#111820] p-4">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <h2 className="text-sm font-semibold text-white">Review Coverage</h2>
              <p className="mt-1 text-xs text-slate-400">Operational coverage counts from saved inspections.</p>
            </div>
            <Layers className="h-4 w-4 text-slate-400" />
          </div>
          <div className="h-52 rounded-lg border border-white/10 bg-black/10 p-2">
            {coverageData.length ? <MetricBarChart data={coverageData} /> : <EmptyMini label="Coverage appears after saved inspections" />}
          </div>
          <p className="mt-3 text-xs leading-5 text-slate-400">
            Analytics converts these records into condition/risk trends, severity mix, class balance, and action quality.
          </p>
        </div>

        <div className="enterprise-panel border border-white/10 bg-[#111820] p-4">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-white">Recent Evidence Intake</h2>
            <ImageUp className="h-4 w-4 text-slate-400" />
          </div>
          {recentItems.length ? (
            <div className="grid gap-3 md:grid-cols-3 xl:grid-cols-1 2xl:grid-cols-3">
              {recentItems.map(({ record, defectFrames, firstFrame, preview, detectionCount }) => (
                <button
                  key={record.id}
                  onClick={() => (firstFrame ? onRestoreFrame(firstFrame, record) : onRestoreRecord(record))}
                  className="group overflow-hidden rounded-lg border border-white/10 bg-white/[0.03] text-left transition hover:border-cyan-300/35 hover:bg-white/[0.06]"
                >
                  <div className="relative aspect-video bg-black/35">
                    {preview ? (
                      <div
                        className="h-full w-full bg-cover bg-center"
                        role="img"
                        aria-label={`${record.asset} preview`}
                        style={{ backgroundImage: `url(${preview})` }}
                      />
                    ) : (
                      <div className="grid h-full place-items-center text-[10px] text-slate-500">No preview</div>
                    )}
                    <span className="absolute left-2 top-2 rounded-md border border-white/10 bg-black/60 px-2 py-1 text-[11px] font-semibold text-slate-100">
                      {record.source}
                    </span>
                  </div>
                  <div className="p-3">
                    <p className="truncate text-sm font-semibold text-white">{record.asset}</p>
                    <p className="mt-1 text-xs text-slate-400">
                      {record.source === "video" ? `${defectFrames.length} defect frames` : `${detectionCount} detections`}
                    </p>
                    <p className="mt-1 truncate text-xs text-slate-500">{record.createdAt}</p>
                  </div>
                </button>
              ))}
            </div>
          ) : (
            <EmptyMini label="Recent evidence appears after image, video, or live monitoring results" />
          )}
          {latest.latencyMs > 0 ? (
            <p className="mt-3 text-xs text-slate-400">Latest active result latency: {formatMs(latest.latencyMs)}</p>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function ArchiveSummaryPanel({
  records,
  detections,
  critical,
  immediate,
  condition,
}: {
  records: number;
  detections: number;
  critical: number;
  immediate: number;
  condition: number;
}) {
  const items = [
    { label: "Saved records", value: records, icon: History, tone: "text-cyan-200" },
    { label: "Stored detections", value: detections, icon: ImageUp, tone: "text-teal-200" },
    { label: "Critical findings", value: critical, icon: ShieldAlert, tone: "text-rose-200" },
    { label: "Immediate actions", value: immediate, icon: AlertTriangle, tone: "text-amber-200" },
    { label: "Avg condition", value: condition.toFixed(1), icon: Gauge, tone: scoreTone(condition) },
  ];
  return (
    <div className="enterprise-panel border border-white/10 bg-[#111820] p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-white">Archive Summary</h2>
        <History className="h-4 w-4 text-slate-400" />
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        {items.map((item) => (
          <div key={item.label} className="rounded-lg border border-white/10 bg-white/[0.03] p-3">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-[11px] uppercase tracking-[0.14em] text-slate-500">{item.label}</p>
                <p className={`mt-1 text-lg font-semibold ${item.tone}`}>{item.value}</p>
              </div>
              <item.icon className="h-4 w-4 text-slate-500" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function OperationalTagPanel({
  actionData,
  conditionData,
  recommendationMetrics,
}: {
  actionData: { name: string; value: number; color: string }[];
  conditionData: { name: string; value: number; color: string }[];
  recommendationMetrics: { immediate: number; planned: number; monitor: number; manualReview: number };
}) {
  return (
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(300px,0.72fr)]">
      <div className="enterprise-panel border border-white/10 bg-[#111820] p-4">
        <div className="mb-3 flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-white">Action Tags and Condition Flags</h2>
            <p className="mt-1 text-xs text-slate-400">
              Recommended-action tags are grouped by urgency. Condition flags count records by health band.
            </p>
          </div>
          <AlertTriangle className="h-4 w-4 text-slate-400" />
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <div className="mb-2 flex items-center justify-between text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">
              Action workload
              <span className="text-slate-300">{actionData.reduce((sum, item) => sum + item.value, 0)}</span>
            </div>
            <div className="h-48 rounded-lg border border-white/10 bg-black/10 p-2">
              {actionData.length ? <MetricBarChart data={actionData} /> : <EmptyMini label="Action tags appear after detections" />}
            </div>
          </div>
          <div>
            <div className="mb-2 flex items-center justify-between text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">
              Condition flags
              <span className="text-slate-300">{conditionData.reduce((sum, item) => sum + item.value, 0)}</span>
            </div>
            <div className="h-48 rounded-lg border border-white/10 bg-black/10 p-2">
              {conditionData.length ? <MetricBarChart data={conditionData} /> : <EmptyMini label="Condition flags appear after saved records" />}
            </div>
          </div>
        </div>
      </div>
      <div className="enterprise-panel border border-white/10 bg-[#111820] p-4">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-white">Recommendation Quality</h2>
          <ShieldAlert className="h-4 w-4 text-slate-400" />
        </div>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-1">
          <TelemetryChip icon={ShieldAlert} label="Immediate" value={String(recommendationMetrics.immediate)} tone="text-rose-200" />
          <TelemetryChip icon={AlertTriangle} label="Planned" value={String(recommendationMetrics.planned)} tone="text-amber-200" />
          <TelemetryChip icon={CircleDot} label="Monitor" value={String(recommendationMetrics.monitor)} tone="text-teal-200" />
          <TelemetryChip icon={RefreshCcw} label="Review needed" value={String(recommendationMetrics.manualReview)} tone="text-violet-200" />
        </div>
        <p className="mt-3 text-xs leading-5 text-slate-400">
          Review needed counts uncertain, unknown, or low-confidence detections so the team can separate model triage from engineering sign-off.
        </p>
      </div>
    </div>
  );
}

function HistoryVisualGallery({
  history,
  onRestoreRecord,
  onRestoreFrame,
}: {
  history: InspectionRecord[];
  onRestoreRecord: (record: InspectionRecord) => void;
  onRestoreFrame: (frame: VideoFrameSummary, record: InspectionRecord) => void;
}) {
  const galleryItems = history.slice(0, 6).map((record) => {
    const defectFrames = record.frames?.filter((frame) => frame.detections > 0) ?? [];
    const firstFrame = defectFrames.find((frame) => frame.evidenceImage || frame.annotatedImage);
    const preview = previewForRecord(record);
    return { record, defectFrames, firstFrame, preview };
  });

  return (
    <div className="enterprise-panel border border-white/10 bg-[#111820] p-4">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-white">Evidence Gallery</h2>
          <p className="mt-1 text-xs text-slate-400">Open recent inspection evidence visually, then drill into frames or focused review.</p>
        </div>
        <ImageUp className="h-4 w-4 text-slate-400" />
      </div>
      {galleryItems.length ? (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2">
          {galleryItems.map(({ record, defectFrames, firstFrame, preview }) => {
            const primaryClasses = Array.from(new Set(detectionsForRecord(record).map((item) => item.className))).slice(0, 3);
            const openFrame = firstFrame && record.source === "video";
            return (
              <button
                key={record.id}
                onClick={() => (openFrame ? onRestoreFrame(firstFrame, record) : onRestoreRecord(record))}
                className="group overflow-hidden rounded-lg border border-white/10 bg-white/[0.03] text-left transition hover:border-cyan-300/35 hover:bg-white/[0.06]"
              >
                <div className="relative aspect-video bg-black/30">
                  {preview ? (
                    <div
                      role="img"
                      aria-label={`${record.asset} inspection evidence`}
                      className="h-full w-full bg-cover bg-center"
                      style={{ backgroundImage: `url(${preview})` }}
                    />
                  ) : (
                    <div className="grid h-full place-items-center text-xs text-slate-500">No stored preview</div>
                  )}
                  <span className={`absolute left-2 top-2 rounded-md border bg-black/55 px-2 py-1 text-xs font-semibold ${scoreTone(record.conditionIndex)}`}>
                    CI {record.conditionIndex.toFixed(0)}
                  </span>
                  <span className="absolute bottom-2 right-2 rounded-md border border-white/10 bg-black/60 px-2 py-1 text-xs text-slate-200">
                    {record.source === "video" ? `${defectFrames.length} defect frames` : `${record.detections.length} boxes`}
                  </span>
                </div>
                <div className="p-3">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <p className="truncate text-sm font-semibold text-white">{record.asset}</p>
                      <p className="mt-1 text-xs text-slate-400">{record.createdAt}</p>
                    </div>
                    <span className="shrink-0 rounded-md border border-cyan-300/25 bg-cyan-300/10 px-2 py-1 text-[11px] font-semibold text-cyan-100">
                      Open
                    </span>
                  </div>
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {primaryClasses.length ? (
                      primaryClasses.map((name) => (
                        <span key={name} className="rounded-full border border-white/10 bg-white/[0.04] px-2 py-0.5 text-[11px] text-slate-300">
                          {name}
                        </span>
                      ))
                    ) : (
                      <span className="rounded-full border border-white/10 bg-white/[0.04] px-2 py-0.5 text-[11px] text-slate-400">
                        no detections
                      </span>
                    )}
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      ) : (
        <EmptyMini label="Evidence thumbnails appear after saved image, video, or live-monitor results" />
      )}
    </div>
  );
}

function HistoryPanel({
  history,
  expandedHistoryId,
  onToggleExpanded,
  onRestoreRecord,
  onRestoreFrame,
  large = false,
}: {
  history: InspectionRecord[];
  expandedHistoryId: string;
  onToggleExpanded: (id: string) => void;
  onRestoreRecord: (record: InspectionRecord) => void;
  onRestoreFrame: (frame: VideoFrameSummary, record: InspectionRecord) => void;
  large?: boolean;
}) {
  return (
    <div className="enterprise-panel border border-white/10 bg-[#111820] p-4">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-white">Inspection Records</h2>
          <p className="mt-1 text-xs text-slate-400">
            Click Open result to focus one saved inspection. Expand video records to choose a defect frame.
          </p>
        </div>
        <History className="h-4 w-4 text-slate-400" />
      </div>
      <div className={`${large ? "max-h-[620px]" : "max-h-[360px]"} overflow-auto pr-1`}>
        {history.length ? (
          <div className="space-y-2">
            {history.map((record) => {
              const defectFrames = record.frames?.filter((frame) => frame.detections > 0) ?? [];
              const expanded = expandedHistoryId === record.id;
              const recordDetections = detectionsForRecord(record);
              return (
                <div key={record.id} className="rounded-lg border border-white/10 bg-white/[0.03]">
                  <div className="grid grid-cols-[1fr_auto_auto] gap-2 p-3">
                    <button onClick={() => onRestoreRecord(record)} className="min-w-0 text-left">
                      <span className="block truncate text-sm font-semibold text-white">{record.asset}</span>
                      <span className="text-xs text-slate-400">
                        {record.createdAt} | {record.source}
                        {record.source === "video"
                          ? ` | ${record.scannedFrames ?? record.stageMetrics.sampledFrames ?? 1} scanned, ${defectFrames.length} defect frames`
                          : ` | ${record.detections.length} defects`}
                      </span>
                      <span className="mt-2 inline-flex rounded-md border border-cyan-300/25 bg-cyan-300/10 px-2 py-1 text-[11px] font-semibold text-cyan-100">
                        Open result for focused review
                      </span>
                    </button>
                    {record.source === "video" ? (
                      <button
                        className="grid h-9 w-9 place-items-center rounded-md border border-white/10 bg-white/[0.04] text-slate-300 transition hover:bg-white/[0.08]"
                        onClick={() => onToggleExpanded(record.id)}
                        title={expanded ? "Hide defect frames" : "Show defect frames"}
                        aria-label={expanded ? "Hide defect frames" : "Show defect frames"}
                      >
                        {expanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                      </button>
                    ) : (
                      <span />
                    )}
                    <span className={`text-sm font-semibold ${scoreTone(record.conditionIndex)}`}>
                      {record.conditionIndex.toFixed(0)}
                    </span>
                  </div>
                  <div className="border-t border-white/10 px-3 py-2 text-xs text-slate-400">
                    {recordDetections.length} total detection{recordDetections.length === 1 ? "" : "s"} counted in analytics
                  </div>
                  {expanded ? (
                    <div className="border-t border-white/10 px-3 py-2">
                      {defectFrames.length ? (
                        <div className="space-y-1.5">
                          {defectFrames.slice(0, large ? 36 : 12).map((frame) => (
                            <button
                              key={frame.id}
                              onClick={() => onRestoreFrame(frame, record)}
                              className="grid w-full grid-cols-[64px_1fr_auto] items-center gap-2 rounded-md bg-white/[0.04] px-2 py-2 text-left text-xs transition hover:bg-white/[0.08]"
                            >
                              <span className="font-mono text-slate-200">{frame.frameTimeSec.toFixed(1)}s</span>
                              <span className="truncate text-slate-300">{frame.classes.join(", ")}</span>
                              <span className="text-slate-400">{frame.detections} box{frame.detections === 1 ? "" : "es"}</span>
                            </button>
                          ))}
                        </div>
                      ) : (
                        <p className="py-2 text-xs text-slate-400">No defect frames captured in this video inspection.</p>
                      )}
                    </div>
                  ) : null}
                </div>
              );
            })}
          </div>
        ) : (
          <EmptyMini label="Inspection records save here after image, video, or live monitoring results" />
        )}
      </div>
    </div>
  );
}

function TelemetryChip({
  icon: Icon,
  label,
  value,
  tone,
}: {
  icon: LucideIcon;
  label: string;
  value: string;
  tone: string;
}) {
  return (
    <div className="flex min-w-0 items-center gap-2 rounded-lg border border-white/10 bg-black/15 px-2 py-1.5">
      <div className="grid h-7 w-7 shrink-0 place-items-center rounded-md border border-white/10 bg-white/[0.04]">
        <Icon className={`h-3.5 w-3.5 ${tone}`} />
      </div>
      <div className="min-w-0">
        <p className="truncate text-[9px] font-semibold uppercase tracking-[0.12em] text-slate-500">{label}</p>
        <p className={`truncate text-xs font-semibold ${tone}`}>{value}</p>
      </div>
    </div>
  );
}

function Info({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-white/10 bg-white/[0.03] p-3">
      <p className="text-xs uppercase tracking-[0.16em] text-slate-500">{label}</p>
      <p className="mt-1 truncate text-sm font-semibold capitalize text-slate-100">{value}</p>
    </div>
  );
}

function ActionPlanCard({ detection }: { detection: Detection }) {
  const meta = priorityMeta[detection.priority];
  const Icon = meta.icon;
  return (
    <div className="rounded-lg border border-amber-300/20 bg-amber-300/[0.06] p-3">
      <div className="mb-2 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] text-amber-100">
          <Icon className="h-4 w-4" />
          Recommended Action
        </div>
        <span className={`rounded-full border px-2 py-1 text-xs ${priorityClasses[detection.priority]}`}>{meta.label}</span>
      </div>
      <p className="text-sm leading-6 text-slate-200">{detection.action}</p>
      <p className="mt-2 text-xs leading-5 text-slate-400">{meta.description}</p>
    </div>
  );
}

function VideoJobPanel({
  job,
  onCancel,
  onLoadResult,
}: {
  job: VideoJob;
  onCancel: () => void;
  onLoadResult: () => void;
}) {
  const active = job.status === "queued" || job.status === "running";
  const complete = job.status === "completed" && Boolean(job.result);
  const progress = Math.max(0, Math.min(100, Math.round(job.progress || 0)));
  const statusTone =
    job.status === "completed"
      ? "text-teal-200"
      : job.status === "failed" || job.status === "cancelled"
        ? "text-rose-200"
        : "text-violet-200";

  return (
    <div className="enterprise-panel overflow-hidden border border-violet-300/20 bg-[#111820]/95 p-4 shadow-2xl shadow-black/40 backdrop-blur">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <Video className="h-4 w-4 text-violet-200" />
            <p className="text-sm font-semibold text-white">Background Video Task</p>
            <span className={`rounded-full border border-white/10 bg-white/[0.04] px-2 py-0.5 text-xs ${statusTone}`}>
              {job.status}
            </span>
          </div>
          <p className="mt-1 truncate text-xs text-slate-400">{job.message || job.stage}</p>
        </div>
        <div className="flex shrink-0 flex-wrap gap-2">
          {active ? (
            <button
              onClick={onCancel}
              className="inline-flex h-9 items-center justify-center rounded-lg border border-rose-300/30 bg-rose-300/10 px-3 text-xs font-semibold text-rose-100 hover:bg-rose-300/15"
            >
              Cancel
            </button>
          ) : null}
          {complete ? (
            <button
              onClick={onLoadResult}
              className="theme-secondary-button inline-flex h-9 items-center justify-center rounded-lg bg-violet-300 px-3 text-xs font-semibold text-slate-950 hover:bg-violet-200"
            >
              Load result
            </button>
          ) : null}
        </div>
      </div>
      <div className="mt-3 h-2 overflow-hidden rounded-full bg-black/30">
        <div className="h-full rounded-full bg-violet-300 transition-all" style={{ width: `${progress}%` }} />
      </div>
      <div className="mt-3 grid gap-2 text-xs md:grid-cols-4">
        <span className="rounded-md border border-white/10 bg-white/[0.04] px-2 py-2 text-slate-300">
          progress<br />
          <strong className="text-white">{progress}%</strong>
        </span>
        <span className="rounded-md border border-white/10 bg-white/[0.04] px-2 py-2 text-slate-300">
          frames<br />
          <strong className="text-white">
            {job.processedFrames ?? 0}/{job.totalFrames ?? "?"}
          </strong>
        </span>
        <span className="rounded-md border border-white/10 bg-white/[0.04] px-2 py-2 text-slate-300">
          positives<br />
          <strong className="text-white">{job.positiveFrames ?? 0}</strong>
        </span>
        <span className="rounded-md border border-white/10 bg-white/[0.04] px-2 py-2 text-slate-300">
          queue<br />
          <strong className="text-white">{job.queuePosition && job.queuePosition > 0 ? `#${job.queuePosition}` : job.stage}</strong>
        </span>
      </div>
    </div>
  );
}

function NoDetectionNotice({
  confidence,
  onReviewSensitivity,
}: {
  confidence: number;
  onReviewSensitivity: () => void;
}) {
  return (
    <div className="rounded-lg border border-amber-300/25 bg-amber-300/10 p-4">
      <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-amber-100">
        <AlertTriangle className="h-4 w-4" />
        No Stage 1 boxes above threshold
      </div>
      <p className="text-sm leading-6 text-amber-50/80">
        The detector did run, but it found no defect boxes above confidence {Math.round(confidence * 100)}%.
        Multi-defect review lowers the threshold to 10% and keeps NMS permissive enough for separated candidates.
      </p>
      <button
        className="mt-3 inline-flex h-9 items-center rounded-lg border border-amber-200/30 bg-amber-200/10 px-3 text-sm font-semibold text-amber-50 hover:bg-amber-200/15"
        onClick={onReviewSensitivity}
      >
        Enable multi-defect review
      </button>
    </div>
  );
}

function EmptyMini({ label }: { label: string }) {
  return (
    <div className="grid h-full min-h-32 place-items-center rounded-lg border border-dashed border-white/10 bg-white/[0.02] p-5 text-center text-sm text-slate-500">
      {label}
    </div>
  );
}
