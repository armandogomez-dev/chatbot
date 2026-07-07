import type { ChatResponse, UserInfo } from "../types";

const BASE = "/api";

export async function sendMessage(message: string, userInfo?: UserInfo): Promise<ChatResponse> {
  const res = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, user_info: userInfo }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? "Error al conectar con el servidor.");
  }
  return res.json();
}
