"use client";

import { useState } from "react";
import AnalyzeForm from "@/components/AnalyzeForm";
import ResultCard from "@/components/ResultCard";
import XAIPanel from "@/components/XAIPanel";
import HistoryPanel from "@/components/HistoryPanel";
import { AnalyzeResponse } from "@/lib/api";
import { addHistory, HistoryEntry } from "@/lib/history";

export default function Home() {
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [currentText, setCurrentText] = useState("");
  const [loading, setLoading] = useState(false);
  const [historyKey, setHistoryKey] = useState(0);

  function handleResult(text: string, res: AnalyzeResponse) {
    setResult(res);
    setCurrentText(text);
    addHistory({ text, result: res, timestamp: Date.now() });
    setHistoryKey((k) => k + 1);
  }

  function handleHistorySelect(entry: HistoryEntry) {
    setResult(entry.result);
    setCurrentText(entry.text);
  }

  return (
    <main className="max-w-5xl mx-auto px-4 sm:px-6 py-10">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-stone-800 tracking-tight">
          Harmful Content Detection
        </h1>
        <p className="text-sm text-stone-500 mt-1">
          Multi-label risk classifier — transformer + character n-grams + safety rules
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-6">
          <AnalyzeForm onResult={handleResult} loading={loading} setLoading={setLoading} />
          <HistoryPanel onSelect={handleHistorySelect} refreshKey={historyKey} />
        </div>

        <div className="space-y-6">
          {result ? (
            <>
              <ResultCard result={result} />
              <XAIPanel key={currentText} text={currentText} />
            </>
          ) : (
            <div className="flex items-center justify-center h-48 border border-dashed border-stone-200 rounded-lg">
              <p className="text-sm text-stone-400">Analysis results will appear here</p>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
