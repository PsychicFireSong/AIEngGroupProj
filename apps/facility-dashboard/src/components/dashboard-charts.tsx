"use client";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
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

export function ConditionTrendChart({ data }: { data: TrendDatum[] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data}>
        <defs>
          <linearGradient id="condition" x1="0" x2="0" y1="0" y2="1">
            <stop offset="5%" stopColor="#2dd4bf" stopOpacity={0.45} />
            <stop offset="95%" stopColor="#2dd4bf" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="rgba(255,255,255,0.08)" />
        <XAxis dataKey="label" tick={{ fill: "#94a3b8", fontSize: 12 }} />
        <YAxis domain={[0, 100]} tick={{ fill: "#94a3b8", fontSize: 12 }} />
        <Tooltip contentStyle={tooltipStyle} />
        <Area type="monotone" dataKey="condition" stroke="#2dd4bf" fill="url(#condition)" strokeWidth={2} />
      </AreaChart>
    </ResponsiveContainer>
  );
}
