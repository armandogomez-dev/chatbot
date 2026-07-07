export function ChatHeader() {
  return (
    <header className="flex items-center gap-3 border-b border-slate-200 bg-white px-6 py-4 shadow-sm">
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
    </header>
  );
}
