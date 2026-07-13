import { Badge } from "../../atoms/Badge";
import { WhatsAppAlert } from "../WhatsAppAlert/WhatsAppAlert";
import type { Message } from "../../../types";

interface Props {
  message: Message;
}

export function MessageBubble({ message }: Props) {
  if (message.role === "system") {
    return (
      <div className="flex justify-center">
        <div className="flex items-center gap-2 rounded-full bg-emerald-50 border border-emerald-200 px-4 py-1.5">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 shrink-0" />
          <span className="text-[11px] text-emerald-700 font-medium">{message.content}</span>
        </div>
      </div>
    );
  }

  const isUser = message.role === "user";
  const showSpecialistAlert = !isUser && message.specialistAlert === true;

  return (
    <div className={`flex flex-col gap-1 ${isUser ? "items-end" : "items-start"}`}>
      {!showSpecialistAlert && (
        <div
          className={`max-w-[75%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
            isUser
              ? "rounded-br-sm bg-primary-600 text-white"
              : "rounded-bl-sm bg-white text-slate-800 shadow-sm"
          }`}
        >
          {message.content}
        </div>
      )}
      {!isUser && message.riskLevel && message.riskConfidence !== undefined && (
        <Badge level={message.riskLevel} confidence={message.riskConfidence} />
      )}
      {showSpecialistAlert && <WhatsAppAlert />}
      <span className="text-[10px] text-slate-400">
        {message.timestamp.toLocaleTimeString("es-CO", {
          hour: "2-digit",
          minute: "2-digit",
        })}
      </span>
    </div>
  );
}
