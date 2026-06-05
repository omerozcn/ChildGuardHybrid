"use client";

import { useEffect, useState } from "react";
import { health, HealthModel } from "@/lib/api";
import { Shield } from "lucide-react";

export default function Navbar() {
  const [models, setModels] = useState<HealthModel[]>([]);
  const [status, setStatus] = useState<"ok" | "degraded" | "loading">("loading");

  useEffect(() => {
    health()
      .then((r) => { setModels(r.models); setStatus(r.status as "ok" | "degraded"); })
      .catch(() => setStatus("degraded"));
  }, []);

  return (
    <header className="border-b border-stone-200 bg-white sticky top-0 z-50">
      <div className="max-w-5xl mx-auto px-6 h-14 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Shield className="w-5 h-5 text-stone-700" />
          <span className="font-semibold text-stone-800 tracking-tight">ChildGuard AI</span>
          <span className="text-xs text-stone-400 ml-1">v2.0</span>
        </div>
        <div className="flex items-center gap-3">
          {models.map((m) => (
            <div key={m.name} className="flex items-center gap-1.5 text-xs text-stone-500">
              <span className={`w-1.5 h-1.5 rounded-full ${m.loaded ? "bg-emerald-500" : "bg-red-400"}`} />
              {m.name}
            </div>
          ))}
          {status === "loading" && <span className="text-xs text-stone-400">Checking...</span>}
        </div>
      </div>
    </header>
  );
}
