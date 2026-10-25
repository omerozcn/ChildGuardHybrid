const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface CategoryResult {
  name: string;
  display_name: string;
  score: number;
  flagged: boolean;
  threshold: number;
}

export interface RuleHit {
  name: string;
  description: string;
  match: string;
}

export type RiskLevel = "low" | "medium" | "high";

export interface AnalyzeResponse {
  harmful: boolean;
  risk_level: RiskLevel;
  score: number;
  categories: CategoryResult[];
  targets_minor: CategoryResult;
  rule_hits: RuleHit[];
  language_warning: boolean;
  model_version: string;
}

export interface TokenWeight {
  token: string;
  weight: number;
}

export interface ExplainResponse {
  label: string;
  base_score: number;
  words: TokenWeight[];
  ngrams: TokenWeight[];
}

export interface HealthResponse {
  status: "ok" | "degraded";
  model_version: string | null;
  base_model: string | null;
  uses_transformer: boolean;
  device: string | null;
  error: string | null;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = await res.text();
    try { detail = JSON.parse(detail).detail ?? detail; } catch {}
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

export function analyze(text: string): Promise<AnalyzeResponse> {
  return post("/api/analyze", { text });
}

export function explain(text: string, label = "toxicity"): Promise<ExplainResponse> {
  return post("/api/explain", { text, label });
}

export async function health(): Promise<HealthResponse> {
  const res = await fetch(`${API_URL}/api/health`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
