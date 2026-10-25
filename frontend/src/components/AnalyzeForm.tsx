"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { analyze, AnalyzeResponse } from "@/lib/api";

const EXAMPLES = [
  "You are a loser, nobody likes you in this school.",
  "nobody would even notice if you were gone",
  "aw, is the little baby going to cry now?",
  "ur such a l0ser lol",
  "My son killed it at his piano recital!",
];

interface Props {
  onResult: (text: string, result: AnalyzeResponse) => void;
  loading: boolean;
  setLoading: (v: boolean) => void;
}

export default function AnalyzeForm({ onResult, loading, setLoading }: Props) {
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit() {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      onResult(text, await analyze(text));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card className="border-stone-200 shadow-none">
      <CardHeader className="pb-3">
        <CardTitle className="text-base font-medium text-stone-700">Text Analysis</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <textarea
          className="w-full border border-stone-200 rounded-md p-3 text-sm text-stone-800 resize-none focus:outline-none focus:ring-1 focus:ring-stone-400 placeholder:text-stone-400"
          rows={4}
          maxLength={5000}
          placeholder="Enter an English message to analyze..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) handleSubmit();
          }}
        />

        {error && (
          <p className="text-xs text-red-600 bg-red-50 border border-red-200 rounded-md px-3 py-2">
            {error}
          </p>
        )}

        <div className="flex gap-2">
          <Button
            onClick={handleSubmit}
            disabled={loading || !text.trim()}
            className="bg-stone-800 hover:bg-stone-700 text-white text-sm"
          >
            {loading ? "Analyzing..." : "Analyze"}
          </Button>
          <Button variant="outline" onClick={() => setText("")} className="text-sm border-stone-200">
            Clear
          </Button>
        </div>

        <div className="pt-1">
          <p className="text-xs text-stone-400 mb-2">Examples</p>
          <div className="flex flex-col gap-1">
            {EXAMPLES.map((ex, i) => (
              <button
                key={i}
                onClick={() => setText(ex)}
                className="text-left text-xs text-stone-500 hover:text-stone-800 truncate transition-colors"
              >
                → {ex}
              </button>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
