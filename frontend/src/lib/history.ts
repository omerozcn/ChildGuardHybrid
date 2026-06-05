import { AnalyzeResponse } from "./api";

export interface HistoryEntry {
  id: string;
  text: string;
  age_group: string;
  result: AnalyzeResponse;
  timestamp: number;
}

const KEY = "childguard_history";

export function getHistory(): HistoryEntry[] {
  if (typeof window === "undefined") return [];
  try {
    return JSON.parse(localStorage.getItem(KEY) || "[]");
  } catch {
    return [];
  }
}

export function addHistory(entry: Omit<HistoryEntry, "id">): HistoryEntry {
  const item: HistoryEntry = { ...entry, id: crypto.randomUUID() };
  const list = [item, ...getHistory()].slice(0, 50);
  localStorage.setItem(KEY, JSON.stringify(list));
  return item;
}

export function deleteHistory(id: string): void {
  const list = getHistory().filter((e) => e.id !== id);
  localStorage.setItem(KEY, JSON.stringify(list));
}

export function clearHistory(): void {
  localStorage.removeItem(KEY);
}
