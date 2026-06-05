"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const EXAMPLES = [
  "You are a loser, nobody likes you in this school.",
  "You're so incredibly stupid, stop talking forever.",
  "I love playing football with my friends after school.",
  "Look at yourself, you are absolutely disgusting.",
];

const AGE_GROUPS = ["Younger", "Pre-Teen", "Teen"];

interface Props {
  onResult: (text: string, ageGroup: string, result: unknown) => void;
  onAgeChange: (age: string) => void;
  loading: boolean;
  setLoading: (v: boolean) => void;
}

export default function AnalyzeForm({ onResult, onAgeChange, loading, setLoading }: Props) {
  const [text, setText] = useState("");
  const [ageGroup, setAgeGroup] = useState("Teen");

  async function handleSubmit() {
    if (!text.trim()) return;
    setLoading(true);
    try {
      const { analyze } = await import("@/lib/api");
      const result = await analyze(text, ageGroup);
      onResult(text, ageGroup, result);
    } catch (e) {
      alert("Error: " + e);
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
          placeholder="Enter a message to analyze..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />

        <div className="flex items-center gap-3">
          <label className="text-xs text-stone-500 font-medium">Age Group</label>
          <div className="flex gap-1">
            {AGE_GROUPS.map((g) => (
              <button
                key={g}
                onClick={() => { setAgeGroup(g); onAgeChange(g); }}
                className={`px-3 py-1 text-xs rounded-md border transition-colors ${
                  ageGroup === g
                    ? "bg-stone-800 text-white border-stone-800"
                    : "bg-white text-stone-600 border-stone-200 hover:border-stone-400"
                }`}
              >
                {g}
              </button>
            ))}
          </div>
        </div>

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
