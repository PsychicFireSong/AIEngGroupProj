"use client";

import {
  Bar,
  BarChart,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";

type SeverityDatum = {
  name: string;
  value: number;
  color: string;
};

type DefectDatum = {
  name: string;
  value: number;
};

type MetricDatum = {
  name: string;
  value: number;
  color: string;
};

type TrendDatum = {
  label: string;
  condition: number;
  risk: number;
};

const tooltipStyle = {
  background: "#111820",
  border: "1px solid rgba(255,255,255,0.12)",
};

// ── Types for new analytics panels ───────────────────────────────────────────

export type ClassSeverityRow = {
  className: string;
  minor: number;
  moderate: number;
  critical: number;
  uncertain: number;
};

export type ClassConfidenceRow = {
  className: string;
  avgDetConf: number;   // Stage 1 average confidence (0-1)
  avgSevConf: number;   // Stage 2 average severity confidence (0-1)
  count: number;
  uncertainRate: number; // fraction of detections that are uncertain/unknown
};

export type AreaCoverageRow = {
  className: string;
  avgArea: number;   // average areaRatio (0-1)
  maxArea: number;
  count: number;
};

export function SeverityPieChart({ data }: { data: SeverityDatum[] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <PieChart>
        <Pie innerRadius={52} outerRadius={78} data={data} dataKey="value" nameKey="name">
          {data.map((item) => (
            <Cell key={item.name} fill={item.color} />
          ))}
        </Pie>
        <Tooltip contentStyle={tooltipStyle} />
      </PieChart>
    </ResponsiveContainer>
  );
}

export function DefectBarChart({ data }: { data: DefectDatum[] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} layout="vertical">
        <CartesianGrid stroke="rgba(255,255,255,0.08)" horizontal={false} />
        <XAxis type="number" hide />
        <YAxis type="category" dataKey="name" width={96} tick={{ fill: "#cbd5e1", fontSize: 11 }} />
        <Tooltip contentStyle={tooltipStyle} />
        <Bar dataKey="value" fill="#38bdf8" radius={[0, 6, 6, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function MetricBarChart({ data }: { data: MetricDatum[] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} layout="vertical" margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
        <CartesianGrid stroke="rgba(255,255,255,0.08)" horizontal={false} />
        <XAxis type="number" hide allowDecimals={false} />
        <YAxis type="category" dataKey="name" width={108} tick={{ fill: "#cbd5e1", fontSize: 11 }} />
        <Tooltip contentStyle={tooltipStyle} />
        <Bar dataKey="value" radius={[0, 6, 6, 0]}>
          {data.map((item) => (
            <Cell key={item.name} fill={item.color} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ── Class × Severity cross-table ─────────────────────────────────────────────
const SEV_COLS: { key: keyof Omit<ClassSeverityRow, "className">; label: string; color: string }[] = [
  { key: "critical",  label: "Critical",  color: "#fb7185" },
  { key: "moderate",  label: "Moderate",  color: "#facc15" },
  { key: "minor",     label: "Minor",     color: "#2dd4bf" },
  { key: "uncertain", label: "Uncertain", color: "#a78bfa" },
];

export function ClassSeverityTable({ rows }: { rows: ClassSeverityRow[] }) {
  if (!rows.length) return (
    <p className="py-6 text-center text-xs text-slate-500">No detections yet</p>
  );
  const total = (r: ClassSeverityRow) => r.critical + r.moderate + r.minor + r.uncertain;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-white/10">
            <th className="py-2 pr-4 text-left font-semibold text-slate-400">Defect class</th>
            {SEV_COLS.map((c) => (
              <th key={c.key} className="px-3 py-2 text-center font-semibold" style={{ color: c.color }}>
                {c.label}
              </th>
            ))}
            <th className="px-3 py-2 text-center font-semibold text-slate-400">Total</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const t = total(row);
            return (
              <tr key={row.className} className="border-b border-white/5 hover:bg-white/[0.03]">
                <td className="py-2 pr-4 font-medium text-slate-200 capitalize">{row.className.replace(/_/g, " ")}</td>
                {SEV_COLS.map((c) => {
                  const v = row[c.key];
                  return (
                    <td key={c.key} className="px-3 py-2 text-center">
                      {v > 0 ? (
                        <span className="inline-flex items-center justify-center rounded px-2 py-0.5 font-semibold" style={{ color: c.color, background: `${c.color}18` }}>
                          {v}
                        </span>
                      ) : (
                        <span className="text-slate-600">—</span>
                      )}
                    </td>
                  );
                })}
                <td className="px-3 py-2 text-center font-semibold text-slate-300">{t}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ── Per-class confidence & uncertainty breakdown ───────────────────────────────
function confidenceLabel(v: number) {
  if (v >= 0.8) return { text: "Very confident", color: "#2dd4bf" };
  if (v >= 0.6) return { text: "Fairly confident", color: "#86efac" };
  if (v >= 0.4) return { text: "Moderate", color: "#facc15" };
  return { text: "Low confidence", color: "#fb7185" };
}

export function ClassConfidencePanel({ rows }: { rows: ClassConfidenceRow[] }) {
  if (!rows.length) return (
    <p className="py-6 text-center text-xs text-slate-500">No detections yet</p>
  );
  return (
    <div className="space-y-2">
      {rows.map((row) => {
        const overallConf = (row.avgDetConf + row.avgSevConf) / 2;
        const label = confidenceLabel(overallConf);
        const reviewCount = Math.round(row.uncertainRate * row.count);
        return (
          <div key={row.className} className="rounded-lg border border-white/[0.07] bg-white/[0.03] px-4 py-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-slate-200 capitalize">{row.className.replace(/_/g, " ")}</span>
              <span className="text-xs" style={{ color: label.color }}>{label.text}</span>
            </div>
            <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-white/10">
              <div className="h-full rounded-full transition-all" style={{ width: `${overallConf * 100}%`, background: label.color }} />
            </div>
            <div className="mt-2 flex items-center justify-between text-xs text-slate-500">
              <span>{row.count} detection{row.count !== 1 ? "s" : ""}</span>
              {reviewCount > 0 ? (
                <span className="text-amber-400">{reviewCount} may need a second look</span>
              ) : (
                <span className="text-slate-600">All clear</span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── Area coverage per class ───────────────────────────────────────────────────
export function AreaCoverageChart({ rows }: { rows: AreaCoverageRow[] }) {
  if (!rows.length) return (
    <p className="py-6 text-center text-xs text-slate-500">No detections yet</p>
  );
  const data = rows.map((r) => ({
    name: r.className.replace(/_/g, " "),
    avg: parseFloat((r.avgArea * 100).toFixed(1)),
    max: parseFloat((r.maxArea * 100).toFixed(1)),
    count: r.count,
  }));
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 48, bottom: 4, left: 4 }}>
        <CartesianGrid stroke="rgba(255,255,255,0.07)" horizontal={false} />
        <XAxis
          type="number"
          tickFormatter={(v) => `${v}%`}
          tick={{ fill: "#94a3b8", fontSize: 11 }}
          domain={[0, "dataMax + 5"]}
        />
        <YAxis type="category" dataKey="name" width={110} tick={{ fill: "#cbd5e1", fontSize: 11 }} />
        <Tooltip
          contentStyle={tooltipStyle}
          formatter={(value, name) => [`${value}%`, name === "avg" ? "Avg area" : "Max area"]}
        />
        <Bar dataKey="avg" name="avg" fill="#38bdf8" radius={[0, 4, 4, 0]} label={{ position: "right", fill: "#94a3b8", fontSize: 11, formatter: (v: unknown) => `${v}%` }} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function ConditionTrendChart({ data }: { data: TrendDatum[] }) {
  const chartData = data.length ? data : [{ label: "Now", condition: 100, risk: 0 }];
  const width = 640;
  const height = 260;
  const padding = { top: 18, right: 28, bottom: 34, left: 38 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const xFor = (index: number) => padding.left + (chartData.length <= 1 ? plotWidth / 2 : (index / (chartData.length - 1)) * plotWidth);
  const yFor = (value: number) => padding.top + (1 - Math.max(0, Math.min(100, value)) / 100) * plotHeight;
  const pathFor = (key: "condition" | "risk") =>
    chartData
      .map((item, index) => `${index === 0 ? "M" : "L"} ${xFor(index).toFixed(2)} ${yFor(item[key]).toFixed(2)}`)
      .join(" ");
  const areaFor = (key: "condition" | "risk") => {
    const line = pathFor(key);
    const firstX = xFor(0).toFixed(2);
    const lastX = xFor(chartData.length - 1).toFixed(2);
    const baseY = (padding.top + plotHeight).toFixed(2);
    return `${line} L ${lastX} ${baseY} L ${firstX} ${baseY} Z`;
  };

  return (
    <svg className="h-full w-full overflow-visible" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Condition and risk trend chart">
      <defs>
        <linearGradient id="condition-area" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="#2dd4bf" stopOpacity="0.34" />
          <stop offset="100%" stopColor="#2dd4bf" stopOpacity="0" />
        </linearGradient>
        <linearGradient id="risk-area" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.28" />
          <stop offset="100%" stopColor="#f59e0b" stopOpacity="0" />
        </linearGradient>
      </defs>
      {[0, 25, 50, 75, 100].map((tick) => {
        const y = yFor(tick);
        return (
          <g key={tick}>
            <line x1={padding.left} x2={width - padding.right} y1={y} y2={y} stroke="rgba(255,255,255,0.08)" />
            <text x={padding.left - 12} y={y + 4} textAnchor="end" className="fill-slate-500 text-[11px]">
              {tick}
            </text>
          </g>
        );
      })}
      <path d={areaFor("risk")} fill="url(#risk-area)" />
      <path d={areaFor("condition")} fill="url(#condition-area)" />
      <path d={pathFor("risk")} fill="none" stroke="#f59e0b" strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" />
      <path d={pathFor("condition")} fill="none" stroke="#2dd4bf" strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" />
      {chartData.map((item, index) => (
        <g key={`${item.label}-${index}`}>
          <circle cx={xFor(index)} cy={yFor(item.risk)} r="4" fill="#f59e0b" />
          <circle cx={xFor(index)} cy={yFor(item.condition)} r="4" fill="#2dd4bf" />
          <text x={xFor(index)} y={height - 12} textAnchor="middle" className="fill-slate-400 text-[11px]">
            {item.label}
          </text>
        </g>
      ))}
    </svg>
  );
}
