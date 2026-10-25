import { AnalyzeResponse } from "./api";

export interface HistoryEntry {
  id: string;
  text: string;
  result: AnalyzeResponse;
  timestamp: number;
}

// v4 yanıt biçimi farklı olduğu için eski (v3) geçmiş kayıtları ayrı anahtarda kalır.
const KEY = "childguard_history_v4";

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
  try {
    localStorage.setItem(KEY, JSON.stringify(list));
  } catch {}
  return item;
}

export function deleteHistory(id: string): void {
  const list = getHistory().filter((e) => e.id !== id);
  try {
    localStorage.setItem(KEY, JSON.stringify(list));
  } catch {}
}

export function clearHistory(): void {
  try {
    localStorage.removeItem(KEY);
  } catch {}
}
