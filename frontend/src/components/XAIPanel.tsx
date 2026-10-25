"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { explain, ExplainResponse, TokenWeight } from "@/lib/api";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

const LABELS: { value: string; label: string }[] = [
  { value: "toxicity", label: "Harmful" },
  { value: "insult", label: "Insult" },
  { value: "threat", label: "Threat" },
  { value: "identity_hate", label: "Identity hate" },
  { value: "sexual", label: "Sexual" },
];

function BarChartView({ features }: { features: TokenWeight[] }) {
  const sorted = [...features].sort((a, b) => b.weight - a.weight);
  return (
    <ResponsiveContainer width="100%" height={Math.max(120, sorted.length * 24)}>
      <BarChart data={sorted} layout="vertical" margin={{ left: 70, right: 20 }}>
        <XAxis type="number" tick={{ fontSize: 10 }} />
        <YAxis type="category" dataKey="token" tick={{ fontSize: 11 }} width={70} />
        <Tooltip formatter={(v) => Number(v).toFixed(4)} />
        <Bar dataKey="weight" radius={[0, 3, 3, 0]}>
          {sorted.map((f, i) => (
            <Cell key={i} fill={f.weight > 0 ? "#ef4444" : "#60a5fa"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function HighlightView({ words }: { words: TokenWeight[] }) {
  const maxAbs = Math.max(...words.map((w) => Math.abs(w.weight)), 0.001);
  return (
    <p className="text-sm leading-7 text-stone-700">
      {words.map((w, i) => {
        const intensity = Math.abs(w.weight) / maxAbs;
        const bg = w.weight > 0
          ? `rgba(239,68,68,${intensity * 0.45})`
          : `rgba(96,165,250,${intensity * 0.45})`;
        return (
          <span key={i}>
            <span style={{ backgroundColor: bg }} className="rounded px-0.5" title={w.weight.toFixed(4)}>
              {w.token}
            </span>{" "}
          </span>
        );
      })}
    </p>
  );
}

export default function XAIPanel({ text }: { text: string }) {
  const [label, setLabel] = useState("toxicity");
  const [data, setData] = useState<ExplainResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function load(l: string) {
    setLabel(l);
    setLoading(true);
    setError(null);
    try {
      setData(await explain(text, l));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card className="border-stone-200 shadow-none">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <CardTitle className="text-base font-medium text-stone-700">Why this result?</CardTitle>
          <div className="flex gap-1 flex-wrap">
            {LABELS.map((l) => (
              <button
                key={l.value}
                onClick={() => load(l.value)}
                disabled={loading}
                className={`text-xs px-2 py-0.5 rounded border disabled:opacity-50 ${
                  data && label === l.value
                    ? "bg-stone-800 text-white border-stone-800"
                    : "border-stone-200 text-stone-500 hover:border-stone-400"
                }`}
              >
                {l.label}
              </button>
            ))}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-5">
        {!data && !loading && !error && (
          <p className="text-xs text-stone-400">Pick a category to see which words drove the score.</p>
        )}
        {loading && <div className="h-24 bg-stone-100 rounded animate-pulse" />}
        {error && <p className="text-xs text-red-600">{error}</p>}
        {data && !loading && (
          <>
            <div className="space-y-2">
              <p className="text-xs font-medium text-stone-600">
                Word impact — score drop when each word is removed
              </p>
              <HighlightView words={data.words} />
              <BarChartView features={data.words.filter((w) => Math.abs(w.weight) > 1e-4)} />
            </div>
            <div className="border-t border-stone-100" />
            <div className="space-y-2">
              <p className="text-xs font-medium text-stone-600">
                Character / word patterns (lexical model, exact contributions)
              </p>
              <BarChartView features={data.ngrams} />
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
