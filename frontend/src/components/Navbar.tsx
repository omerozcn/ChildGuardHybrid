"use client";

import { useEffect, useState } from "react";
import { health, HealthResponse } from "@/lib/api";
import { Shield } from "lucide-react";

export default function Navbar() {
  const [info, setInfo] = useState<HealthResponse | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    health().then(setInfo).catch(() => setFailed(true));
  }, []);

  const ok = info?.status === "ok";
  const label = failed
    ? "Backend unreachable"
    : !info
      ? "Checking..."
      : ok
        ? `${info.base_model ?? "lexical-only"} · ${info.device}`
        : info.error ?? "Model not loaded";

  return (
    <header className="border-b border-stone-200 bg-white sticky top-0 z-50">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Shield className="w-5 h-5 text-stone-700" />
          <span className="font-semibold text-stone-800 tracking-tight">ChildGuard AI</span>
          <span className="text-xs text-stone-400 ml-1">v{info?.model_version ?? "4"}</span>
        </div>
        <div className="flex items-center gap-1.5 text-xs text-stone-500 min-w-0" title={label}>
          <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${ok ? "bg-emerald-500" : info || failed ? "bg-red-400" : "bg-stone-300"}`} />
          <span className="truncate">{label}</span>
        </div>
      </div>
    </header>
  );
}
