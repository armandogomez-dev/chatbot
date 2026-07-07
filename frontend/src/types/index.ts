export type RiskLevel = "riesgo" | "no riesgo";

export interface UserInfo {
  name: string;
  phone: string;
  email: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  riskLevel?: RiskLevel;
  riskConfidence?: number;
  timestamp: Date;
}

export interface ChatResponse {
  response: string;
  risk_label: string;
  risk_confidence: number;
}
