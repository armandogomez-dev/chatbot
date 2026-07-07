# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chatbox de apoyo (tesis de maestría) que combina dos modelos Hugging Face en un pipeline:

- **modelo salo** — RoBERTa clasificador binario: `"riesgo"` / `"no riesgo"`
- **modelo jhon** — T5 generador de respuestas (seq2seq, fine-tuned)

## Architecture

```
chatbox/
├── modelo jhon/          # T5ForConditionalGeneration (generador)
├── modelo salo/          # RobertaForSequenceClassification (clasificador)
├── backend/              # FastAPI (Python)
│   ├── main.py           # App + endpoint POST /chat, GET /health
│   ├── inference.py      # Pipeline: classify() → generate()
│   ├── schemas.py        # Pydantic: ChatRequest, ChatResponse
│   └── requirements.txt
└── frontend/             # React 18 + Vite + Tailwind (Atomic Design)
    └── src/
        ├── components/
        │   ├── atoms/        Button, Input, Badge, Spinner
        │   ├── molecules/    MessageBubble, ChatInput
        │   ├── organisms/    ChatHeader, MessageList
        │   ├── templates/    ChatLayout
        │   └── pages/        ChatPage  ← estado global del chat
        ├── services/api.ts   # fetch POST /api/chat
        └── types/index.ts    # Message, ChatResponse, RiskLevel
```

## Inference Pipeline

```
user message
    ↓
modelo salo (RoBERTa) → "riesgo" | "no riesgo"
    ↓
modelo jhon (T5) → genera respuesta con prefix según riesgo
    ↓
ChatResponse { response, risk_label, risk_confidence }
```

El prefix del T5 se configura con variables de entorno:
- `T5_PREFIX_RISK` (default: `"riesgo: "`)
- `T5_PREFIX_NORMAL` (default: `"chat: "`)

Ajustar estos prefijos al formato exacto del fine-tuning.

## Dev Commands

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
npm run build      # producción → dist/
```

El Vite proxy redirige `/api/*` → `http://localhost:8000/*`, por lo que no se necesita CORS en desarrollo.
