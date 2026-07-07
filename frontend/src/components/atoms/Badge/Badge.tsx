import type { RiskLevel } from "../../../types";

interface Props {
  level: RiskLevel;
  confidence: number;
}

export function Badge({ level, confidence }: Props) {
  const isRisk = level === "riesgo";
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${
        isRisk
          ? "bg-red-100 text-red-700"
          : "bg-emerald-100 text-emerald-700"
      }`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${isRisk ? "bg-red-500" : "bg-emerald-500"}`} />
      {isRisk ? "Riesgo" : "Sin riesgo"} · {Math.round(confidence * 100)}%
    </span>
  );
}
