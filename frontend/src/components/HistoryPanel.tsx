"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { getHistory, deleteHistory, clearHistory, HistoryEntry } from "@/lib/history";

interface Props {
  onSelect: (entry: HistoryEntry) => void;
  refreshKey: number;
}

export default function HistoryPanel({ onSelect, refreshKey }: Props) {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);

  useEffect(() => {
    setEntries(getHistory());
  }, [refreshKey]);

  function handleDelete(id: string, e: React.MouseEvent) {
    e.stopPropagation();
    deleteHistory(id);
    setEntries(getHistory());
  }

  function handleClear() {
    clearHistory();
    setEntries([]);
  }

  if (entries.length === 0) return null;

  return (
    <Card className="border-stone-200 shadow-none">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base font-medium text-stone-700">History</CardTitle>
          <button onClick={handleClear} className="text-xs text-stone-400 hover:text-stone-600">Clear all</button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-1 max-h-64 overflow-y-auto">
          {entries.map((e) => {
            const isRisky = e.result.final_score > 0.45;
            return (
              <div
                key={e.id}
                onClick={() => onSelect(e)}
                className="flex items-center justify-between px-3 py-2 rounded-md hover:bg-stone-50 cursor-pointer group"
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${isRisky ? "bg-red-400" : "bg-emerald-400"}`} />
                  <span className="text-xs text-stone-600 truncate">{e.text}</span>
                </div>
                <div className="flex items-center gap-2 flex-shrink-0 ml-2">
                  <span className="text-xs font-mono text-stone-400">{(e.result.final_score * 100).toFixed(0)}%</span>
                  <button
                    onClick={(ev) => handleDelete(e.id, ev)}
                    className="text-stone-300 hover:text-stone-500 opacity-0 group-hover:opacity-100 text-xs"
                  >
                    ×
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
