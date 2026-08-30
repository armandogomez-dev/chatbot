import { type FormEvent, useState } from "react";
import { Button } from "../../atoms/Button";
import { Input } from "../../atoms/Input";
import { Spinner } from "../../atoms/Spinner";

interface Props {
  onSend: (message: string) => void;
  loading: boolean;
  disabled?: boolean;
}

export function ChatInput({ onSend, loading, disabled = false }: Props) {
  const [value, setValue] = useState("");

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = value.trim();
    if (!trimmed || loading || disabled) return;
    onSend(trimmed);
    setValue("");
  }

  return (
    <form onSubmit={handleSubmit} className="flex items-center gap-2">
      <Input
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder={disabled ? "Chat remitido a un profesional" : "Escribe un mensaje..."}
        disabled={loading || disabled}
        autoFocus
      />
      <Button type="submit" disabled={loading || disabled || !value.trim()} className="shrink-0">
        {loading ? <Spinner /> : "Enviar"}
      </Button>
    </form>
  );
}
