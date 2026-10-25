"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AnalyzeResponse, CategoryResult, RiskLevel } from "@/lib/api";

const RISK_STYLE: Record<RiskLevel, { label: string; badge: string; bar: string }> = {
  high: { label: "HIGH RISK", badge: "bg-red-50 text-red-700 border-red-200", bar: "bg-red-500" },
  medium: { label: "HARMFUL", badge: "bg-amber-50 text-amber-700 border-amber-200", bar: "bg-amber-400" },
  low: { label: "SAFE", badge: "bg-emerald-50 text-emerald-700 border-emerald-200", bar: "bg-emerald-400" },
};

const CATEGORY_LABEL: Record<string, string> = {
  insult: "Insult / humiliation",
  threat: "Threat / self-harm",
  sexual: "Sexual content",
  identity_hate: "Identity hate",
  profanity: "Profanity",
  targets_minor: "Targets a minor (experimental)",
};

function ScoreBar({ c }: { c: CategoryResult }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className={c.flagged ? "text-stone-800 font-medium" : "text-stone-500"}>
          {CATEGORY_LABEL[c.name] ?? c.name}
        </span>
        <span className="font-mono text-stone-500">{(c.score * 100).toFixed(1)}%</span>
      </div>
      <div className="relative h-1.5 bg-stone-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${c.flagged ? "bg-red-400" : "bg-stone-300"}`}
          style={{ width: `${c.score * 100}%` }}
        />
        <div
          className="absolute top-0 h-full w-px bg-stone-500"
          style={{ left: `${c.threshold * 100}%` }}
          title={`Threshold ${(c.threshold * 100).toFixed(0)}%`}
        />
      </div>
    </div>
  );
}

export default function ResultCard({ result }: { result: AnalyzeResponse }) {
  const style = RISK_STYLE[result.risk_level];

  return (
    <Card className="border-stone-200 shadow-none">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base font-medium text-stone-700">Result</CardTitle>
          <Badge className={`text-xs font-medium ${style.badge}`} variant="outline">
            {style.label}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <div className="text-3xl font-mono font-semibold text-stone-800">
            {(result.score * 100).toFixed(1)}
            <span className="text-lg text-stone-400">%</span>
          </div>
          <p className="text-xs text-stone-400 mt-1">Probability that the message is harmful</p>
        </div>

        {result.language_warning && (
          <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-md px-3 py-2">
            This message does not look like English. The model is trained on English text, so
            this result is unreliable.
          </p>
        )}

        {result.rule_hits.length > 0 && (
          <div className="text-xs bg-red-50 border border-red-200 rounded-md px-3 py-2 space-y-1">
            <p className="font-medium text-red-700">Safety rule triggered</p>
            {result.rule_hits.map((h) => (
              <p key={h.name} className="text-red-600">
                “{h.match}” — {h.description}
              </p>
            ))}
          </div>
        )}

        <div className="space-y-2.5">
          {result.categories.map((c) => <ScoreBar key={c.name} c={c} />)}
        </div>

        <div className="pt-2 border-t border-stone-100">
          <ScoreBar c={result.targets_minor} />
        </div>

        <p className="text-[11px] text-stone-400">
          Vertical lines mark each category’s decision threshold. Model v{result.model_version}
        </p>
      </CardContent>
    </Card>
  );
}
