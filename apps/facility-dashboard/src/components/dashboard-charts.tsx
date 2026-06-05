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
