import { useEffect, useRef } from "react";
import { MessageBubble } from "../../molecules/MessageBubble";
import { Spinner } from "../../atoms/Spinner";
import type { Message } from "../../../types";

interface Props {
  messages: Message[];
  loading: boolean;
}

export function MessageList({ messages, loading }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  return (
    <div className="flex flex-1 flex-col gap-4 overflow-y-auto px-6 py-4">
      {messages.length === 0 && (
        <p className="m-auto text-sm text-slate-400">
          Escribe algo para comenzar la conversación.
        </p>
      )}
      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}
      {loading && (
        <div className="flex items-center gap-2 text-slate-400">
          <Spinner />
          <span className="text-xs">Pensando...</span>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}
