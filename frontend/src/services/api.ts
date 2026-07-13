import type { ChatResponse, RiskEntry, UserInfo } from "../types";

const BASE = "/api";

export async function sendMessage(
  message: string,
  userInfo: UserInfo | undefined,
  history: RiskEntry[],
  alertSent: boolean,
): Promise<ChatResponse> {
  const res = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, user_info: userInfo, history, alert_sent: alertSent }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? "Error al conectar con el servidor.");
  }
  return res.json();
}
