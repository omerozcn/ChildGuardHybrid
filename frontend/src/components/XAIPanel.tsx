"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { explainShap, explainLime, FeatureWeight } from "@/lib/api";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

function SkeletonLoader({ label }: { label: string }) {
  return (
    <div className="space-y-2 animate-pulse">
      <p className="text-xs text-stone-400">{label}</p>
      {[100, 80, 65, 55, 45].map((w, i) => (
        <div key={i} className="flex items-center gap-2">
          <div className="h-2 bg-stone-200 rounded" style={{ width: `${w * 0.4}px` }} />
          <div className="h-2 bg-stone-100 rounded flex-1" style={{ maxWidth: `${w}px` }} />
        </div>
      ))}
    </div>
  );
}

interface Props {
  text: string;
  ageGroup: string;
}

function BarChartView({ features }: { features: FeatureWeight[] }) {
  const sorted = [...features].sort((a, b) => b.weight - a.weight);
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={sorted} layout="vertical" margin={{ left: 60, right: 20 }}>
        <XAxis type="number" tick={{ fontSize: 10 }} />
        <YAxis type="category" dataKey="token" tick={{ fontSize: 11 }} width={60} />
        <Tooltip formatter={(v: number) => v.toFixed(4)} />
        <Bar dataKey="weight" radius={[0, 3, 3, 0]}>
          {sorted.map((f, i) => (
            <Cell key={i} fill={f.weight > 0 ? "#ef4444" : "#60a5fa"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function HighlightView({ text, features }: { text: string; features: FeatureWeight[] }) {
  const maxAbs = Math.max(...features.map((f) => Math.abs(f.weight)), 0.001);
  const tokenMap = Object.fromEntries(features.map((f) => [f.token.toLowerCase(), f.weight]));

  const words = text.split(/(\s+)/);
  return (
    <p className="text-sm leading-7 text-stone-700">
      {words.map((word, i) => {
        const clean = word.toLowerCase().replace(/[^a-z0-9]/g, "");
        const w = tokenMap[clean];
        if (!w) return <span key={i}>{word}</span>;
        const intensity = Math.abs(w) / maxAbs;
        const bg = w > 0
          ? `rgba(239,68,68,${intensity * 0.4})`
          : `rgba(96,165,250,${intensity * 0.4})`;
        return (
          <span key={i} style={{ backgroundColor: bg }} className="rounded px-0.5 transition-all" title={`${w.toFixed(4)}`}>
            {word}
          </span>
        );
      })}
    </p>
  );
}

function ExplainSection({ title, text, ageGroup, fetchFn }: {
  title: string;
  text: string;
  ageGroup: string;
  fetchFn: () => Promise<FeatureWeight[]>;
}) {
  const [features, setFeatures] = useState<FeatureWeight[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [view, setView] = useState<"chart" | "highlight">("chart");

  async function load() {
    setLoading(true);
    try {
      setFeatures(await fetchFn());
    } catch (e) {
      alert("Error: " + e);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-stone-600">{title}</span>
        <div className="flex gap-1">
          {features && (
            <>
              <button onClick={() => setView("chart")} className={`text-xs px-2 py-0.5 rounded border ${view === "chart" ? "bg-stone-800 text-white border-stone-800" : "border-stone-200 text-stone-500"}`}>Chart</button>
              <button onClick={() => setView("highlight")} className={`text-xs px-2 py-0.5 rounded border ${view === "highlight" ? "bg-stone-800 text-white border-stone-800" : "border-stone-200 text-stone-500"}`}>Highlight</button>
            </>
          )}
          <button onClick={load} disabled={loading} className="text-xs px-2 py-0.5 rounded border border-stone-200 text-stone-500 hover:border-stone-400 disabled:opacity-50">
            {loading ? "Loading..." : features ? "Refresh" : "Explain"}
          </button>
        </div>
      </div>
      {loading && title.startsWith("LIME") && (
        <SkeletonLoader label="Computing LIME explanations..." />
      )}
      {!loading && features && (
        view === "chart"
          ? <BarChartView features={features} />
          : <HighlightView text={text} features={features} />
      )}
    </div>
  );
}

export default function XAIPanel({ text, ageGroup }: Props) {
  return (
    <Card className="border-stone-200 shadow-none">
      <CardHeader className="pb-3">
        <CardTitle className="text-base font-medium text-stone-700">Explainability</CardTitle>
      </CardHeader>
      <CardContent className="space-y-5">
        <ExplainSection
          title="SHAP — XGBoost"
          text={text}
          ageGroup={ageGroup}
          fetchFn={async () => (await explainShap(text, ageGroup)).features}
        />
        <div className="border-t border-stone-100" />
        <ExplainSection
          title="LIME — BERT"
          text={text}
          ageGroup={ageGroup}
          fetchFn={async () => (await explainLime(text, ageGroup)).features}
        />
      </CardContent>
    </Card>
  );
}
