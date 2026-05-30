"use client";

import {
  Activity,
  AlertTriangle,
  Camera,
  CheckCircle2,
  CircleDot,
  Cpu,
  Gauge,
  History,
  ImageUp,
  Layers,
  LayoutDashboard,
  Pause,
  Play,
  RefreshCcw,
  Server,
  Settings2,
  ShieldAlert,
  Upload,
  Video,
  Zap,
  type LucideIcon,
} from "lucide-react";
import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const SeverityPieChart = dynamic(
  () => import("@/components/dashboard-charts").then((mod) => mod.SeverityPieChart),
  { ssr: false },
);
const DefectBarChart = dynamic(
  () => import("@/components/dashboard-charts").then((mod) => mod.DefectBarChart),
  { ssr: false },
);
const ConditionTrendChart = dynamic(
  () => import("@/components/dashboard-charts").then((mod) => mod.ConditionTrendChart),
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

type InferenceResult = {
  detections: Detection[];
  latencyMs: number;
  stageMetrics: {
    stage1LatencyMs: number;
    stage2LatencyMs: number;
    cropsClassified: number;
    detectorModel: string;
    severityModel: string;
  };
  conditionIndex: number;
  riskScore: number;
  annotatedImage?: string;
};

type InspectionRecord = InferenceResult & {
  id: string;
  createdAt: string;
  source: "camera" | "image" | "video";
  asset: string;
  preview?: string;
};

type ModelOption = {
  label: string;
  value: string;
  kind: "detector" | "severity";
};

type Settings = {
  mode: "Near real-time" | "Balanced" | "High accuracy";
  confidence: number;
  iou: number;
  frameIntervalMs: number;
  detectorPath: string;
  severityPath: string;
};

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

const balancedSettings: Omit<Settings, "detectorPath" | "severityPath"> = {
  mode: "Balanced",
  confidence: 0.45,
  iou: 0.45,
  frameIntervalMs: 900,
};

const modeSettings: Record<Settings["mode"], Omit<Settings, "mode" | "detectorPath" | "severityPath">> = {
  "Near real-time": {
    confidence: 0.38,
    iou: 0.45,
    frameIntervalMs: 500,
  },
  Balanced: {
    confidence: 0.45,
    iou: 0.45,
    frameIntervalMs: 900,
  },
  "High accuracy": {
    confidence: 0.52,
    iou: 0.38,
    frameIntervalMs: 1500,
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

const navItems: { label: string; icon: LucideIcon }[] = [
  { label: "Live evidence", icon: Camera },
  { label: "Analytics", icon: Activity },
  { label: "History", icon: History },
  { label: "Settings", icon: Settings2 },
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

function modelDisplayName(value: string) {
  if (!value) return "not selected";
  return value.split(/[\\/]/).pop() || value;
}

function preferredModelValue(models: ModelOption[], kind: ModelOption["kind"]) {
  const candidates = models.filter((item) => item.kind === kind && item.value);
  const preferredTerms = kind === "detector" ? ["defect_detector.pt", "stage1_defect_detector"] : ["severity_cls.pt", "stage2_severity"];
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

function emptyInferenceResult(): InferenceResult {
  return {
    detections: [],
    latencyMs: 0,
    stageMetrics: emptyStageMetrics,
    conditionIndex: 100,
    riskScore: 0,
  };
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const intervalRef = useRef<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [apiOnline, setApiOnline] = useState(false);
  const [apiMessage, setApiMessage] = useState("Checking inference API");
  const [models, setModels] = useState<ModelOption[]>(fallbackModels);
  const [settings, setSettings] = useState<Settings>({
    ...balancedSettings,
    detectorPath: "",
    severityPath: "",
  });
  const [showTuning, setShowTuning] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraMessage, setCameraMessage] = useState("Camera preview is off.");
  const [cameraError, setCameraError] = useState("");
  const [scanning, setScanning] = useState(false);
  const [busy, setBusy] = useState(false);
  const [sourceType, setSourceType] = useState<InspectionRecord["source"]>("camera");
  const [imagePreview, setImagePreview] = useState<string>("");
  const [latest, setLatest] = useState<InferenceResult>(() => emptyInferenceResult());
  const [selectedDetectionId, setSelectedDetectionId] = useState<string>("");
  const [history, setHistory] = useState<InspectionRecord[]>(() => {
    if (typeof window === "undefined") return [];
    const stored = window.localStorage.getItem("facility-inspection-history");
    if (!stored) return [];
    try {
      return (JSON.parse(stored) as InspectionRecord[]).map((record) => normalizeResult(record));
    } catch {
      return [];
    }
  });

  const detectorModels = models.filter((model) => model.kind === "detector");
  const severityModels = models.filter((model) => model.kind === "severity");
  const selectedDetection = latest.detections.find((item) => item.id === selectedDetectionId) ?? latest.detections[0];

  const severityData = useMemo(() => {
    const counts = latest.detections.reduce<Record<Severity, number>>(
      (acc, detection) => {
        acc[detection.severity] += 1;
        return acc;
      },
      { minor: 0, moderate: 0, critical: 0, uncertain: 0, unknown: 0 },
    );
    return Object.entries(counts)
      .filter(([, value]) => value > 0)
      .map(([name, value]) => ({ name, value, color: severityColors[name as Severity] }));
  }, [latest.detections]);

  const defectData = useMemo(() => {
    const counts = latest.detections.reduce<Record<string, number>>((acc, detection) => {
      acc[detection.className] = (acc[detection.className] ?? 0) + 1;
      return acc;
    }, {});
    return Object.entries(counts).map(([name, value]) => ({ name, value }));
  }, [latest.detections]);

  const trendData = useMemo(() => {
    if (history.length === 0) {
      return [
        { label: "T-4", condition: 100, risk: 0 },
        { label: "T-3", condition: 98, risk: 2 },
        { label: "T-2", condition: 96, risk: 4 },
        { label: "T-1", condition: latest.conditionIndex, risk: latest.riskScore },
      ];
    }
    return history.slice(0, 8).reverse().map((item, index) => ({
      label: `S${index + 1}`,
      condition: item.conditionIndex,
      risk: item.riskScore,
    }));
  }, [history, latest.conditionIndex, latest.riskScore]);

  const criticalCount = latest.detections.filter((item) => item.severity === "critical").length;
  const immediateCount = latest.detections.filter((item) => item.priority === "Immediate").length;

  useEffect(() => {
    window.localStorage.setItem("facility-inspection-history", JSON.stringify(history.slice(0, 24)));
  }, [history]);

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

  const stopCamera = useCallback(() => {
    const stream = videoRef.current?.srcObject as MediaStream | null;
    stream?.getTracks().forEach((track) => track.stop());
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    setScanning(false);
    setCameraMessage("Camera preview is off.");
  }, []);

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        window.clearInterval(intervalRef.current);
      }
      stopCamera();
    };
  }, [stopCamera]);

  const startCamera = async () => {
    setSourceType("camera");
    setImagePreview("");
    setLatest(emptyInferenceResult());
    setSelectedDetectionId("");
    setCameraError("");
    setCameraMessage("Requesting camera access...");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => undefined);
      }
      setCameraActive(true);
      setCameraMessage("Camera preview active. Use Live scan to run repeated inference.");
      setApiMessage(apiOnline ? "Camera ready" : apiMessage);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Camera permission was blocked or no camera is available.";
      setCameraActive(false);
      setCameraError(message);
      setCameraMessage("Camera could not be opened.");
      setApiMessage("Camera permission was blocked or no camera is available.");
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

  const captureCameraBlob = async () => {
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

  const recordResult = (result: InferenceResult, source: InspectionRecord["source"], preview?: string) => {
    const record: InspectionRecord = {
      ...result,
      id: crypto.randomUUID(),
      createdAt: new Date().toLocaleString(),
      source,
      asset: "Main Building / External Wall",
      preview,
    };
    setHistory((current) => [record, ...current].slice(0, 24));
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
      recordResult(payload, source, payload.annotatedImage ?? preview);
      setApiMessage(`Inference complete in ${formatMs(payload.latencyMs)}`);
    } catch {
      setApiMessage("Inference failed. Check model paths and backend logs.");
    } finally {
      setBusy(false);
    }
  };

  const runCameraOnce = async () => {
    const blob = await captureCameraBlob();
    if (blob) {
      await inferBlob(blob, "camera");
    }
  };

  const toggleScanning = async () => {
    if (!cameraActive) {
      await startCamera();
    }
    if (scanning) {
      if (intervalRef.current) {
        window.clearInterval(intervalRef.current);
      }
      intervalRef.current = null;
      setScanning(false);
      return;
    }
    await runCameraOnce();
    intervalRef.current = window.setInterval(runCameraOnce, settings.frameIntervalMs);
    setScanning(true);
  };

  const onUploadImage = async (file: File | undefined) => {
    if (!file) return;
    stopCamera();
    setSourceType("image");
    const preview = URL.createObjectURL(file);
    setImagePreview(preview);
    await inferBlob(file, "image", preview);
  };

  const evidenceImage = latest.annotatedImage || imagePreview;
  const showVideo = sourceType === "camera" && cameraActive && !evidenceImage;

  return (
    <main className="min-h-screen bg-[#0b0f14] text-slate-100">
      <div className="grid min-h-screen grid-cols-[76px_minmax(0,1fr)]">
        <aside className="hidden border-r border-white/10 bg-[#11161d] px-3 py-5 md:block">
          <div className="mb-9 grid h-11 w-11 place-items-center rounded-lg border border-cyan-300/30 bg-cyan-300/10">
            <LayoutDashboard className="h-5 w-5 text-cyan-200" />
          </div>
          <nav className="flex flex-col gap-3">
            {navItems.map(({ label, icon: Icon }) => (
              <button
                key={label}
                className="grid h-11 w-11 place-items-center rounded-lg border border-white/10 bg-white/[0.03] text-slate-300 transition hover:border-cyan-300/40 hover:bg-cyan-300/10 hover:text-cyan-100"
                title={label}
              >
                <Icon className="h-5 w-5" />
              </button>
            ))}
          </nav>
        </aside>

        <section className="grid min-h-screen grid-rows-[auto_minmax(0,1fr)]">
          <header className="border-b border-white/10 bg-[#11161d]/95 px-5 py-4 backdrop-blur">
            <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.25em] text-cyan-200">AI Facilities Operations</p>
                <h1 className="mt-1 text-2xl font-semibold tracking-normal text-white">Defect Detection Control Room</h1>
              </div>

              <div className="grid gap-3 md:grid-cols-[minmax(180px,1fr)_minmax(180px,1fr)_auto_auto] xl:min-w-[720px]">
                <select
                  className="h-11 rounded-lg border border-white/10 bg-[#0d1218] px-3 text-sm text-slate-100 outline-none focus:border-cyan-300/60"
                  value={settings.detectorPath}
                  onChange={(event) => setSettings((current) => ({ ...current, detectorPath: event.target.value }))}
                >
                  {detectorModels.map((model) => (
                    <option key={model.value || model.label} value={model.value}>
                      {model.label}
                    </option>
                  ))}
                </select>

                <select
                  className="h-11 rounded-lg border border-white/10 bg-[#0d1218] px-3 text-sm text-slate-100 outline-none focus:border-cyan-300/60"
                  value={settings.severityPath}
                  onChange={(event) => setSettings((current) => ({ ...current, severityPath: event.target.value }))}
                >
                  {severityModels.map((model) => (
                    <option key={model.value || model.label} value={model.value}>
                      {model.label}
                    </option>
                  ))}
                </select>

                <button
                  className="inline-flex h-11 items-center justify-center gap-2 rounded-lg bg-teal-300 px-4 text-sm font-semibold text-slate-950 transition hover:bg-teal-200"
                  onClick={applyRecommended}
                >
                  <Zap className="h-4 w-4" />
                  Recommended
                </button>

                <button
                  className="inline-flex h-11 items-center justify-center gap-2 rounded-lg border border-white/10 bg-white/[0.04] px-4 text-sm font-semibold text-slate-100 transition hover:bg-white/[0.08]"
                  onClick={() => setShowTuning((value) => !value)}
                >
                  <Settings2 className="h-4 w-4" />
                  Tuning
                </button>
              </div>
            </div>

            {showTuning ? (
              <div className="mt-4 grid gap-4 rounded-lg border border-white/10 bg-[#0b1016] p-4 lg:grid-cols-4">
                <label className="text-sm text-slate-300">
                  <span className="mb-2 block text-xs uppercase tracking-[0.16em] text-slate-500">Mode</span>
                  <select
                    className="h-10 w-full rounded-lg border border-white/10 bg-[#111820] px-3 text-slate-100 outline-none"
                    value={settings.mode}
                    onChange={(event) => setMode(event.target.value as Settings["mode"])}
                  >
                    <option>Near real-time</option>
                    <option>Balanced</option>
                    <option>High accuracy</option>
                  </select>
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
            ) : null}
          </header>

          <div className="grid min-h-0 gap-4 p-4 xl:grid-cols-[320px_minmax(0,1fr)_340px]">
            <section className="order-2 flex min-h-0 flex-col gap-4 xl:order-1">
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
              <MetricPalette
                detections={latest.detections.length}
                critical={criticalCount}
                immediate={immediateCount}
                latency={latest.latencyMs}
                condition={latest.conditionIndex}
              />
              <div className="rounded-lg border border-white/10 bg-[#111820] p-4">
                <div className="mb-3 flex items-center justify-between">
                  <h2 className="text-sm font-semibold text-white">Severity Mix</h2>
                  <Layers className="h-4 w-4 text-slate-400" />
                </div>
                <div className="h-48">
                  {severityData.length ? (
                    <SeverityPieChart data={severityData} />
                  ) : (
                    <EmptyMini label="No defects detected yet" />
                  )}
                </div>
              </div>
            </section>

            <section className="order-1 flex min-h-0 flex-col xl:order-2">
              <div className="flex min-h-0 flex-1 flex-col rounded-lg border border-white/10 bg-[#12171d] shadow-2xl shadow-black/30">
                <div className="flex flex-col gap-3 border-b border-white/10 p-4 lg:flex-row lg:items-center lg:justify-between">
                  <div>
                    <h2 className="text-lg font-semibold text-white">Visual Evidence</h2>
                    <p className="text-sm text-slate-400">Camera, uploaded images, and annotated detections stay in the main workspace.</p>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <button
                      onClick={cameraActive ? stopCamera : startCamera}
                      className="inline-flex h-10 items-center gap-2 rounded-lg border border-cyan-300/30 bg-cyan-300/10 px-3 text-sm font-semibold text-cyan-100 hover:bg-cyan-300/15"
                    >
                      <Camera className="h-4 w-4" />
                      {cameraActive ? "Stop camera" : "Open camera"}
                    </button>
                    <button
                      onClick={toggleScanning}
                      className="inline-flex h-10 items-center gap-2 rounded-lg border border-teal-300/30 bg-teal-300/10 px-3 text-sm font-semibold text-teal-100 hover:bg-teal-300/15"
                    >
                      {scanning ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                      {scanning ? "Pause scan" : "Live scan"}
                    </button>
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="inline-flex h-10 items-center gap-2 rounded-lg border border-white/10 bg-white/[0.04] px-3 text-sm font-semibold text-slate-100 hover:bg-white/[0.08]"
                    >
                      <Upload className="h-4 w-4" />
                      Upload
                    </button>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/*"
                      className="hidden"
                      onChange={(event) => onUploadImage(event.target.files?.[0])}
                    />
                  </div>
                </div>
                <div className="border-b border-white/10 bg-white/[0.02] px-4 py-3 text-xs leading-5 text-slate-400">
                  <strong className="text-slate-200">Open camera</strong> shows the raw preview only.{" "}
                  <strong className="text-slate-200">Live scan</strong> repeatedly captures frames and sends them through Stage 1 detection and Stage 2 severity classification.
                </div>

                <div className="relative min-h-[420px] flex-1 overflow-hidden bg-black lg:min-h-[560px]">
                  {showVideo ? (
                    <video ref={videoRef} className="h-full w-full object-contain" autoPlay muted playsInline />
                  ) : evidenceImage ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={evidenceImage} alt="Inspection evidence" className="h-full w-full object-contain" />
                  ) : (
                    <div className="grid h-full min-h-[420px] place-items-center bg-[radial-gradient(circle_at_center,rgba(45,212,191,0.14),transparent_45%),#05070a]">
                      <div className="max-w-sm text-center">
                        <div className="mx-auto mb-4 grid h-14 w-14 place-items-center rounded-lg border border-cyan-300/30 bg-cyan-300/10">
                          <Video className="h-7 w-7 text-cyan-100" />
                        </div>
                        <h3 className="text-lg font-semibold text-white">
                          {cameraError ? "Camera is not available" : "Open camera or upload evidence"}
                        </h3>
                        <p className="mt-2 text-sm text-slate-400">
                          {cameraError || cameraMessage || "The annotated feed will stay here while the palettes update around it."}
                        </p>
                      </div>
                    </div>
                  )}

                  {latest.detections.map((detection) => (
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
                      onClick={() => setSelectedDetectionId(detection.id)}
                    >
                      <span
                        className="absolute -top-7 left-0 rounded-md px-2 py-1 text-xs font-semibold text-slate-950"
                        style={{ background: severityColors[detection.severity] }}
                      >
                        {detection.className} | {detection.severity}
                      </span>
                    </button>
                  ))}
                </div>

                <div className="grid gap-3 border-t border-white/10 p-4 md:grid-cols-3 2xl:grid-cols-6">
                  <EvidenceStat icon={Gauge} label="Condition" value={latest.conditionIndex.toFixed(1)} tone={scoreTone(latest.conditionIndex)} />
                  <EvidenceStat icon={ShieldAlert} label="Risk Score" value={latest.riskScore.toFixed(1)} tone="text-amber-200" />
                  <EvidenceStat icon={Camera} label="Stage 1" value={formatMs(latest.stageMetrics.stage1LatencyMs)} tone="text-cyan-200" />
                  <EvidenceStat icon={Layers} label="Stage 2" value={formatMs(latest.stageMetrics.stage2LatencyMs)} tone="text-teal-200" />
                  <EvidenceStat icon={Cpu} label="Total" value={formatMs(latest.latencyMs)} tone="text-violet-200" />
                  <EvidenceStat icon={CircleDot} label="Mode" value={settings.mode} tone="text-violet-200" />
                </div>
              </div>
            </section>

            <section className="order-3 flex min-h-0 flex-col gap-4">
              <div className="rounded-lg border border-white/10 bg-[#111820]">
                <div className="flex items-center justify-between border-b border-white/10 p-4">
                  <h2 className="text-sm font-semibold text-white">Detected Defects</h2>
                  <span className="rounded-full bg-white/10 px-2 py-1 text-xs text-slate-300">{latest.detections.length} findings</span>
                </div>
                <div className="max-h-[412px] overflow-auto p-3">
                  {latest.detections.length ? (
                    <div className="space-y-3">
                      {latest.detections.map((detection) => (
                        <button
                          key={detection.id}
                          onClick={() => setSelectedDetectionId(detection.id)}
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
                            <span className={`rounded-full border px-2 py-1 text-xs ${priorityClasses[detection.priority]}`}>
                              {detection.priority}
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
                        </button>
                      ))}
                    </div>
                  ) : latest.latencyMs > 0 ? (
                    <NoDetectionNotice
                      confidence={settings.confidence}
                      onReviewSensitivity={() => {
                        setSettings((current) => ({ ...current, confidence: 0.2, mode: "Near real-time" }));
                        setShowTuning(true);
                      }}
                    />
                  ) : (
                    <EmptyMini label="No active findings" />
                  )}
                </div>
              </div>

              <div className="rounded-lg border border-white/10 bg-[#111820] p-4">
                <div className="mb-3 flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4 text-amber-200" />
                  <h2 className="text-sm font-semibold text-white">Selected Finding</h2>
                </div>
                {selectedDetection ? (
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
                    <p className="rounded-lg border border-white/10 bg-white/[0.04] p-3 text-sm leading-6 text-slate-300">
                      {selectedDetection.action}
                    </p>
                  </div>
                ) : (
                  <p className="text-sm text-slate-400">Select a finding from the visual evidence or defect list.</p>
                )}
              </div>

              <div className="rounded-lg border border-white/10 bg-[#111820] p-4">
                <div className="mb-3 flex items-center justify-between">
                  <h2 className="text-sm font-semibold text-white">Defect Distribution</h2>
                  <RefreshCcw className="h-4 w-4 text-slate-400" />
                </div>
                <div className="h-44">
                  {defectData.length ? (
                    <DefectBarChart data={defectData} />
                  ) : (
                    <EmptyMini label="Distribution appears after inference" />
                  )}
                </div>
              </div>
            </section>

            <section className="order-4 grid gap-4 xl:col-span-3 xl:grid-cols-[minmax(0,1fr)_420px]">
              <div className="rounded-lg border border-white/10 bg-[#111820] p-4">
                <div className="mb-3 flex items-center justify-between">
                  <h2 className="text-sm font-semibold text-white">Condition Index Over Time</h2>
                  <Activity className="h-4 w-4 text-slate-400" />
                </div>
                <div className="h-56">
                  <ConditionTrendChart data={trendData} />
                </div>
              </div>

              <div className="rounded-lg border border-white/10 bg-[#111820] p-4">
                <div className="mb-3 flex items-center justify-between">
                  <h2 className="text-sm font-semibold text-white">Inspection History</h2>
                  <History className="h-4 w-4 text-slate-400" />
                </div>
                <div className="max-h-56 overflow-auto">
                  {history.length ? (
                    <div className="space-y-2">
                      {history.slice(0, 8).map((record) => (
                        <button
                          key={record.id}
                          onClick={() => {
                            setLatest(normalizeResult(record));
                            setSourceType(record.source);
                            setImagePreview(record.preview ?? "");
                            setSelectedDetectionId(record.detections[0]?.id ?? "");
                          }}
                          className="grid w-full grid-cols-[1fr_auto] gap-3 rounded-lg border border-white/10 bg-white/[0.03] p-3 text-left transition hover:bg-white/[0.06]"
                        >
                          <span>
                            <span className="block text-sm font-semibold text-white">{record.asset}</span>
                            <span className="text-xs text-slate-400">
                              {record.createdAt} | {record.source} | {record.detections.length} defects
                            </span>
                          </span>
                          <span className={`text-sm font-semibold ${scoreTone(record.conditionIndex)}`}>
                            {record.conditionIndex.toFixed(0)}
                          </span>
                        </button>
                      ))}
                    </div>
                  ) : (
                    <EmptyMini label="Inspection records save here" />
                  )}
                </div>
              </div>
            </section>
          </div>
        </section>
      </div>
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
}: {
  apiOnline: boolean;
  apiMessage: string;
  busy: boolean;
  scanning: boolean;
}) {
  return (
    <div className="rounded-lg border border-white/10 bg-[#111820] p-4">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-white">System Status</h2>
        <Server className="h-4 w-4 text-slate-400" />
      </div>
      <div className="space-y-3">
        <StatusRow active={apiOnline} label="Inference API" detail={apiMessage} />
        <StatusRow active={scanning} label="Live scanning" detail={scanning ? "Frame sampling enabled" : "Manual capture"} />
        <StatusRow active={!busy} label="Processing" detail={busy ? "Inference running" : "Ready"} />
      </div>
    </div>
  );
}

function StatusRow({ active, label, detail }: { active: boolean; label: string; detail: string }) {
  return (
    <div className="flex gap-3 rounded-lg border border-white/10 bg-white/[0.03] p-3">
      {active ? <CheckCircle2 className="mt-0.5 h-4 w-4 text-teal-300" /> : <AlertTriangle className="mt-0.5 h-4 w-4 text-amber-300" />}
      <div className="min-w-0">
        <p className="text-sm font-semibold text-white">{label}</p>
        <p className="truncate text-xs text-slate-400">{detail}</p>
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
}: {
  stage1Latency: number;
  stage2Latency: number;
  cropsClassified: number;
  detectorModel: string;
  severityModel: string;
}) {
  return (
    <div className="rounded-lg border border-white/10 bg-[#111820] p-4">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-white">Two-Stage Inference</h2>
        <Zap className="h-4 w-4 text-teal-200" />
      </div>
      <div className="space-y-3">
        <StageRow
          accent="cyan"
          icon={Camera}
          title="Stage 1 Detector"
          metric={formatMs(stage1Latency)}
          detail={`${detectorModel} | boxes and defect classes`}
        />
        <div className="ml-5 h-5 border-l border-dashed border-white/20" />
        <StageRow
          accent="teal"
          icon={Layers}
          title="Stage 2 Severity"
          metric={formatMs(stage2Latency)}
          detail={`${severityModel} | ${cropsClassified} cropped regions classified`}
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
}: {
  icon: LucideIcon;
  title: string;
  metric: string;
  detail: string;
  accent: "cyan" | "teal";
}) {
  const color = accent === "cyan" ? "text-cyan-200 bg-cyan-300/10 border-cyan-300/25" : "text-teal-200 bg-teal-300/10 border-teal-300/25";
  return (
    <div className="rounded-lg border border-white/10 bg-white/[0.03] p-3">
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 gap-3">
          <div className={`grid h-9 w-9 shrink-0 place-items-center rounded-lg border ${color}`}>
            <Icon className="h-4 w-4" />
          </div>
          <div className="min-w-0">
            <p className="text-sm font-semibold text-white">{title}</p>
            <p className="truncate text-xs text-slate-400">{detail}</p>
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
    { label: "Detected defects", value: detections, color: "text-cyan-200", icon: ImageUp },
    { label: "Critical severity", value: critical, color: "text-rose-200", icon: ShieldAlert },
    { label: "Immediate work", value: immediate, color: "text-amber-200", icon: AlertTriangle },
    { label: "Condition index", value: condition.toFixed(1), color: scoreTone(condition), icon: Gauge },
    { label: "Last latency", value: formatMs(latency), color: "text-violet-200", icon: Cpu },
  ];

  return (
    <div className="grid gap-3">
      {items.map((item) => (
        <div key={item.label} className="flex items-center justify-between rounded-lg border border-white/10 bg-[#111820] p-4">
          <div>
            <p className="text-xs uppercase tracking-[0.16em] text-slate-500">{item.label}</p>
            <p className={`mt-1 text-2xl font-semibold ${item.color}`}>{item.value}</p>
          </div>
          <item.icon className="h-5 w-5 text-slate-500" />
        </div>
      ))}
    </div>
  );
}

function EvidenceStat({
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
    <div className="rounded-lg border border-white/10 bg-white/[0.03] p-3">
      <div className="mb-2 flex items-center gap-2 text-xs uppercase tracking-[0.16em] text-slate-500">
        <Icon className="h-4 w-4" />
        {label}
      </div>
      <p className={`truncate text-lg font-semibold ${tone}`}>{value}</p>
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
        Lower sensitivity can reveal weak candidates, but may also increase wrong classes.
      </p>
      <button
        className="mt-3 inline-flex h-9 items-center rounded-lg border border-amber-200/30 bg-amber-200/10 px-3 text-sm font-semibold text-amber-50 hover:bg-amber-200/15"
        onClick={onReviewSensitivity}
      >
        Set review sensitivity
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
