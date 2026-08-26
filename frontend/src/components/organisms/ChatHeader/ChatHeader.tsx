type ChatHeaderProps = {
  onNewSession?: () => void;
};

export function ChatHeader({ onNewSession }: ChatHeaderProps) {
  return (
    <header className="flex items-center justify-between gap-3 border-b border-slate-200 bg-white px-6 py-4 shadow-sm">
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-full bg-primary-600 text-white text-sm font-bold">
          IA
        </div>
        <div>
          <p className="text-sm font-semibold text-slate-800">Asistente de apoyo</p>
          <p className="text-xs text-emerald-500 flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 inline-block" />
            En línea
          </p>
        </div>
      </div>
      {onNewSession && (
        <button
          type="button"
          onClick={onNewSession}
          className="rounded-full border border-slate-200 px-3 py-1 text-xs font-medium text-slate-600 transition hover:border-primary-600 hover:text-primary-600"
        >
          Nueva conversación
        </button>
      )}
    </header>
  );
}
