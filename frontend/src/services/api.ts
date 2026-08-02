import type { ChatResponse, RiskEntry, UserInfo } from "../types";

const BASE = "/api";

export async function sendMessage(
  message: string,
  userInfo: UserInfo | undefined,
  history: RiskEntry[],
  alertSent: boolean,
  sessionId?: string,
  newSession = false,
): Promise<ChatResponse> {
  const res = await fetch(`${BASE}/chat`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      user_info: userInfo,
      history,
      alert_sent: alertSent,
      session_id: sessionId ?? null,
      new_session: newSession,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? "Error al conectar con el servidor.");
  }
  return res.json();
}
