const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface AgePrediction {
  predicted: string;
  scores: Record<string, number>;
}

export interface AnalyzeResponse {
  label: string;
  final_score: number;
  bert_score: number;
  xgb_score: number;
  age_group: string;
  age_prediction: AgePrediction | null;
}

export interface FeatureWeight {
  token: string;
  weight: number;
}

export interface ExplainResponse {
  features: FeatureWeight[];
}

export interface HealthModel {
  name: string;
  loaded: boolean;
  path: string;
}

export interface HealthResponse {
  status: string;
  models: HealthModel[];
}

export async function analyze(text: string, age_group: string): Promise<AnalyzeResponse> {
  const res = await fetch(`${API_URL}/api/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, age_group }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function explainShap(text: string, age_group: string): Promise<ExplainResponse> {
  const res = await fetch(`${API_URL}/api/explain/shap`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, age_group }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function explainLime(text: string, age_group: string, num_samples = 500): Promise<ExplainResponse> {
  const res = await fetch(`${API_URL}/api/explain/lime`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, age_group, num_samples }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function health(): Promise<HealthResponse> {
  const res = await fetch(`${API_URL}/api/health`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
