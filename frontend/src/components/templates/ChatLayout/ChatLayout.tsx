import type { ReactNode } from "react";

interface Props {
  header: ReactNode;
  body: ReactNode;
  footer: ReactNode;
}

export function ChatLayout({ header, body, footer }: Props) {
  return (
    <div className="flex h-screen items-center justify-center bg-slate-100">
      <div className="flex h-full w-full max-w-2xl flex-col overflow-hidden rounded-none shadow-xl sm:h-[90vh] sm:rounded-2xl">
        {header}
        {body}
        <div className="border-t border-slate-200 bg-white px-4 py-3">{footer}</div>
      </div>
    </div>
  );
}
