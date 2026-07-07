import { useState } from "react";
import type { UserInfo } from "../../../types";

interface Props {
  onComplete: (info: UserInfo) => void;
}

export function OnboardingModal({ onComplete }: Props) {
  const [name, setName] = useState("");
  const [phone, setPhone] = useState("");
  const [email, setEmail] = useState("");
  const [error, setError] = useState("");

  function normalizePhone(raw: string): string {
    const digits = raw.replace(/\D/g, "");
    return `+57${digits}`;
  }

  function handlePhoneChange(e: React.ChangeEvent<HTMLInputElement>) {
    const digits = e.target.value.replace(/\D/g, "").slice(0, 10);
    setPhone(digits);
    setError("");
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim()) {
      setError("El nombre es obligatorio.");
      return;
    }
    if (phone.length !== 10) {
      setError("El número de WhatsApp debe tener 10 dígitos.");
      return;
    }
    onComplete({ name: name.trim(), phone: normalizePhone(phone), email: email.trim() });
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 backdrop-blur-sm px-4">
      <div className="w-full max-w-sm rounded-2xl bg-white p-6 shadow-xl">
        <div className="mb-5 text-center">
          <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-primary-100 text-primary-600 text-xl font-bold">
            IA
          </div>
          <h2 className="text-base font-semibold text-slate-800">Un momento antes de continuar</h2>
          <p className="mt-1 text-xs text-slate-500 leading-relaxed">
            Comparte tu nombre y WhatsApp para que el especialista pueda contactarte
            si lo necesitas.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col gap-3">
          <div>
            <label className="mb-1 block text-xs font-medium text-slate-700">
              Nombre <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => { setName(e.target.value); setError(""); }}
              placeholder="Tu nombre completo"
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-800 outline-none transition focus:border-primary-500 focus:ring-2 focus:ring-primary-100"
            />
          </div>

          <div>
            <label className="mb-1 block text-xs font-medium text-slate-700">
              Número de WhatsApp <span className="text-red-500">*</span>
            </label>
            <input
              type="tel"
              inputMode="numeric"
              value={phone}
              onChange={handlePhoneChange}
              maxLength={10}
              placeholder="3001234567 (sin +57)"
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-800 outline-none transition focus:border-primary-500 focus:ring-2 focus:ring-primary-100"
            />
            <p className="mt-0.5 text-[10px] text-slate-400">Solo para contactarte si el sistema detecta riesgo.</p>
          </div>

          <div>
            <label className="mb-1 block text-xs font-medium text-slate-700">
              Correo electrónico
              <span className="ml-1 text-slate-400 font-normal">(opcional)</span>
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="tu@correo.com"
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-800 outline-none transition focus:border-primary-500 focus:ring-2 focus:ring-primary-100"
            />
          </div>

          {error && <p className="text-xs text-red-500">{error}</p>}

          <p className="text-[10px] text-slate-400 leading-relaxed">
            Tu información es confidencial y solo se usará para contactarte en caso
            de que el sistema detecte una situación de riesgo.
          </p>

          <button
            type="submit"
            className="mt-1 w-full rounded-xl bg-primary-600 py-2.5 text-sm font-semibold text-white transition hover:bg-primary-700 active:scale-95"
          >
            Comenzar
          </button>
        </form>
      </div>
    </div>
  );
}
