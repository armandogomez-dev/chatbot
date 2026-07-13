import { useState } from "react";
import { ChatLayout } from "../../templates/ChatLayout";
import { ChatHeader } from "../../organisms/ChatHeader";
import { MessageList } from "../../organisms/MessageList";
import { ChatInput } from "../../molecules/ChatInput";
import { OnboardingModal } from "../../molecules/OnboardingModal/OnboardingModal";
import { sendMessage } from "../../../services/api";
import type { Message, RiskEntry, RiskLevel, UserInfo } from "../../../types";

// Debe coincidir con RISK_ALERT_WINDOW en el backend: solo se envían los
// últimos N resultados de riesgo, que es todo lo que el backend evalúa.
const RISK_HISTORY_WINDOW = 5;

function makeId() {
  return Math.random().toString(36).slice(2);
}

export function ChatPage() {
  const [userInfo, setUserInfo] = useState<UserInfo | null>(null);
  const [pendingMessage, setPendingMessage] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [riskHistory, setRiskHistory] = useState<RiskEntry[]>([]);
  const [alertSent, setAlertSent] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function processMessage(text: string, info: UserInfo) {
    setLoading(true);
    setError(null);
    try {
      const data = await sendMessage(text, info, riskHistory, alertSent);
      const justAlerted = data.alert_sent && !alertSent;
      const botMsg: Message = {
        id: makeId(),
        role: "assistant",
        content: data.response,
        riskLevel: data.risk_label as RiskLevel,
        riskConfidence: data.risk_confidence,
        specialistAlert: justAlerted,
        timestamp: new Date(),
      };

      setAlertSent(data.alert_sent);
      setRiskHistory((prev) =>
        [...prev, { risk_label: data.risk_label, risk_confidence: data.risk_confidence }].slice(
          -RISK_HISTORY_WINDOW,
        ),
      );

      setMessages((prev) => {
        const next = [...prev, botMsg];
        if (justAlerted) {
          next.push({
            id: makeId(),
            role: "system",
            content: "Se notificó al especialista con tu información de contacto. Te comunicará pronto.",
            timestamp: new Date(),
          });
        }
        return next;
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error desconocido.");
    } finally {
      setLoading(false);
    }
  }

  async function handleSend(text: string) {
    const userMsg: Message = {
      id: makeId(),
      role: "user",
      content: text,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMsg]);

    if (!userInfo) {
      setPendingMessage(text);
      return;
    }

    await processMessage(text, userInfo);
  }

  async function handleOnboardingComplete(info: UserInfo) {
    setUserInfo(info);
    if (pendingMessage) {
      const msg = pendingMessage;
      setPendingMessage(null);
      await processMessage(msg, info);
    }
  }

  return (
    <>
      {pendingMessage !== null && !userInfo && (
        <OnboardingModal onComplete={handleOnboardingComplete} />
      )}
      <ChatLayout
        header={<ChatHeader />}
        body={<MessageList messages={messages} loading={loading} />}
        footer={
          <div className="flex flex-col gap-1">
            {error && <p className="text-xs text-red-500">{error}</p>}
            <ChatInput onSend={handleSend} loading={loading} />
          </div>
        }
      />
    </>
  );
}
