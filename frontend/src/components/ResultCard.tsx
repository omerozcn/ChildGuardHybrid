"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AnalyzeResponse } from "@/lib/api";

interface Props {
  result: AnalyzeResponse;
}

function ScoreBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-stone-500">
        <span>{label}</span>
        <span className="font-mono">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-1.5 bg-stone-100 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${value * 100}%` }} />
      </div>
    </div>
  );
}

export default function ResultCard({ result }: Props) {
  const isRisky = result.final_score > 0.45;

  return (
    <Card className="border-stone-200 shadow-none">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base font-medium text-stone-700">Result</CardTitle>
          <Badge
            className={`text-xs font-medium ${
              isRisky
                ? "bg-red-50 text-red-700 border-red-200"
                : "bg-emerald-50 text-emerald-700 border-emerald-200"
            }`}
            variant="outline"
          >
            {isRisky ? "HARMFUL" : "SAFE"}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="text-3xl font-mono font-semibold text-stone-800">
          {(result.final_score * 100).toFixed(1)}
          <span className="text-lg text-stone-400">%</span>
        </div>

        <div className="space-y-2.5">
          <ScoreBar label="Hybrid Score" value={result.final_score} color={isRisky ? "bg-red-400" : "bg-emerald-400"} />
          <ScoreBar label="BERT" value={result.bert_score} color="bg-stone-400" />
          <ScoreBar label="XGBoost" value={result.xgb_score} color="bg-stone-300" />
        </div>

        {result.age_prediction && (
          <div className="pt-2 border-t border-stone-100">
            <p className="text-xs text-stone-500 mb-2">Age Group Prediction</p>
            <div className="flex gap-2">
              {Object.entries(result.age_prediction.scores).map(([group, score]) => (
                <div
                  key={group}
                  className={`flex-1 text-center py-2 rounded-md border text-xs ${
                    group === result.age_prediction!.predicted
                      ? "border-stone-800 bg-stone-800 text-white"
                      : "border-stone-200 text-stone-500"
                  }`}
                >
                  <div className="font-medium">{group}</div>
                  <div className="font-mono">{(score * 100).toFixed(0)}%</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
