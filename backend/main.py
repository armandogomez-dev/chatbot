import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Dict, List
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from feedback_store import list_feedback, save_feedback
from inference import inference
from notifier import send_email_alert, send_whatsapp_alert
from schemas import ChatRequest, ChatResponse, FeedbackRequest, UserInfo, RiskEntry

# Token simple para proteger el endpoint de resultados de evaluación (no es para
# autenticar usuarios del chat, solo para que no cualquiera con el link vea las
# observaciones de los psicólogos). Vacío en local = endpoint abierto.
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

logging.basicConfig(level=logging.INFO)

# Umbral de confianza para considerar un mensaje como "alto riesgo" (0.0–1.0)
RISK_ALERT_THRESHOLD = float(os.getenv("RISK_ALERT_THRESHOLD", "0.75"))

# Se dispara la alerta al especialista cuando, de los últimos RISK_ALERT_WINDOW
# mensajes (incluyendo el actual), al menos RISK_ALERT_MIN_COUNT son de alto riesgo.
RISK_ALERT_WINDOW = int(os.getenv("RISK_ALERT_WINDOW", "5"))
RISK_ALERT_MIN_COUNT = int(os.getenv("RISK_ALERT_MIN_COUNT", "3"))

PROFESSIONAL_REQUEST_PHRASE = "necesito hablar con un profesional"
SUPPORT_AGENT_REMINDER = (

    'Recuerda que solo soy un agente de apoyo. Si tienes una emergencia y '
    'requieres hablar con un profesional solo escribe la frase: '
    '"Necesito hablar con un profesional"'
)


def _is_high_risk(label: str, confidence: float) -> bool:
    return label == "riesgo" and confidence >= RISK_ALERT_THRESHOLD


def _normalize_user_text(text: str) -> str:
    normalized = re.sub(r"[^a-záéíóúüñ\s]", " ", text.lower())
    return " ".join(normalized.split())


def _requests_professional_contact(text: str) -> bool:
    return PROFESSIONAL_REQUEST_PHRASE in _normalize_user_text(text)


_GREETING_PREFIX_RE = re.compile(r"^(?:¡?hola|hello|hi)[!,.]?\s*", re.IGNORECASE)


def _strip_repeated_greeting(text: str) -> str:
    """El generador suele abrir cada respuesta con un saludo ('Hola, ...'); a partir
    del segundo turno de la conversación eso se siente repetitivo, así que se recorta."""
    stripped = _GREETING_PREFIX_RE.sub("", text, count=1)
    if not stripped:
        return text
    return stripped[0].upper() + stripped[1:]


# In-memory store for chat histories: session_id -> list of messages
# Each message is a dict: {"role": "user" or "assistant", "content": str, "timestamp": float}
# We'll keep it simple and just store the last N messages (where N is RISK_ALERT_WINDOW * 2?).
# But we want to keep the entire conversation for context? Let's store the last 20 messages.
chat_histories: Dict[str, List[dict]] = {}
blocked_sessions: set[str] = set()
MAX_HISTORY_LENGTH = 20  # keep last 20 messages (10 turns)


@asynccontextmanager
async def lifespan(app: FastAPI):
    inference.load()
    yield


app = FastAPI(title="Chatbox API", lifespan=lifespan)

# En producción (single-service) el frontend se sirve desde el mismo origen que la API,
# así que CORS no hace falta ahí; se mantiene localhost:5173 para `npm run dev` local.
_extra_origins = [o.strip() for o in os.getenv("CORS_EXTRA_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", *_extra_origins],
    allow_methods=["*"],
    allow_headers=["*"],
)

api = APIRouter(prefix="/api")


@api.get("/health")
def health():
    return {"status": "ok"}


@api.post("/chat")
async def chat(request: Request, body: ChatRequest, background_tasks: BackgroundTasks):
    text = body.message.strip()
    if not text:
        raise HTTPException(status_code=422, detail="El mensaje no puede estar vacío.")

    # Determine session ID: from body, cookie, or generate new.
    # If the client explicitly requests a new session, start fresh and clear
    # the prior history for that session if it exists.
    if body.new_session:
        previous_session_id = body.session_id or request.cookies.get("session_id")
        if previous_session_id is not None:
            chat_histories.pop(previous_session_id, None)
            blocked_sessions.discard(previous_session_id)
        session_id = body.session_id or str(uuid4())
    else:
        session_id = body.session_id or request.cookies.get("session_id")
        if session_id is None:
            session_id = str(uuid4())

    if session_id in blocked_sessions or body.alert_sent:
        raise HTTPException(
            status_code=403,
            detail="Este chat ha sido remitido a un profesional y ya no acepta más mensajes.",
        )

    is_first_turn = not chat_histories.get(session_id)

    # Compute risk and sentiment for the current user message (needed for storage, alert and routing)
    text_en = inference._translate_to_en(text)
    risk_label, risk_confidence = inference.classify(text_en)
    # Red de seguridad basada en reglas: el clasificador ML puede fallar en ideación
    # suicida pasiva o poco explícita (ver apply_risk_safety_net). Se aplica sobre el
    # texto original en español, antes de la traducción, para no perder matices.
    risk_label, risk_confidence = inference.apply_risk_safety_net(risk_label, risk_confidence, text)
    is_risk = risk_label == "riesgo"
    sentiment_label, _ = inference.classify_sentiment(text_en)

    # Generate response in English from the current message only (the generators were
    # fine-tuned on single-turn inputs, not multi-turn "Usuario:/Asistente:" transcripts).
    response_en = inference.generate(text_en, is_risk, sentiment_label)
    response_es = inference._translate_to_es(response_en)
    if not is_first_turn:
        response_es = _strip_repeated_greeting(response_es)
    response_es = inference.append_default_closing(response_es, text)
    if is_first_turn:
        response_es = f"{response_es}\n\n{SUPPORT_AGENT_REMINDER}"

    # Prepare user message entry for storage (includes risk info)
    import time
    timestamp = time.time()
    user_message_entry = {
        "role": "user",
        "content": text,
        "risk_label": risk_label,
        "risk_confidence": risk_confidence,
        "timestamp": timestamp
    }

    # Get current history for this session
    current_history = chat_histories.get(session_id, [])

    # Append the user message
    current_history.append(user_message_entry)

    # Prepare assistant message entry (without risk info)
    assistant_message_entry = {
        "role": "assistant",
        "content": response_es,
        "timestamp": time.time()
    }
    current_history.append(assistant_message_entry)

    # Trim history to avoid growing indefinitely (keep last 20 user-assistant turns = 40 messages)
    if len(current_history) > 40:
        current_history = current_history[-40:]
    chat_histories[session_id] = current_history

    # Now compute alert using the stored user messages in the history
    # Extract the user messages from the history (in order)
    user_messages = [entry for entry in current_history if entry["role"] == "user"]
    # Take the last RISK_ALERT_WINDOW user messages (or fewer if not enough)
    recent_user_messages = user_messages[-RISK_ALERT_WINDOW:]
    high_risk_count = sum(1 for msg in recent_user_messages if _is_high_risk(msg["risk_label"], msg["risk_confidence"]))

    direct_professional_request = _requests_professional_contact(text)
    alert_sent = body.alert_sent
    should_alert = direct_professional_request or high_risk_count >= RISK_ALERT_MIN_COUNT
    if not body.alert_sent and should_alert:
        print(
            "ALERT TRIGGERED: "
            f"direct_professional_request={direct_professional_request}, "
            f"high_risk_count={high_risk_count}, threshold={RISK_ALERT_MIN_COUNT}"
        )
        ui = body.user_info
        alert_confidence = 1.0 if direct_professional_request else risk_confidence
        background_tasks.add_task(send_email_alert, text, alert_confidence, ui.name, ui.phone, ui.email)
        background_tasks.add_task(send_whatsapp_alert, text, alert_confidence, ui.name, ui.phone, ui.email)
        alert_sent = True
        blocked_sessions.add(session_id)
    else:
        print(
            "ALERT CHECK: "
            f"direct_professional_request={direct_professional_request}, "
            f"high_risk_count={high_risk_count}, threshold={RISK_ALERT_MIN_COUNT}, "
            f"alert_sent={alert_sent}"
        )

    chat_blocked = session_id in blocked_sessions or alert_sent

    # Prepare JSON response and set cookie
    response_content = {
        "response": response_es,
        "risk_label": risk_label,
        "risk_confidence": risk_confidence,
        "alert_sent": alert_sent,
        "chat_blocked": chat_blocked,
        "session_id": session_id
    }
    response = JSONResponse(content=response_content)
    # Set cookie (httpOnly=False so JS can read if needed; adjust as needed)
    response.set_cookie(key="session_id", value=session_id, httponly=False, max_age=60*60*24*30)  # 30 days
    return response


@api.post("/feedback")
def submit_feedback(body: FeedbackRequest):
    entry_id = save_feedback(body.model_dump())
    return {"id": entry_id}


@api.get("/feedback")
def get_feedback(request: Request):
    if ADMIN_TOKEN and request.headers.get("x-admin-token") != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Token inválido.")
    entries = list_feedback()
    criteria = ["risk_detection", "empathy", "coherence", "clarity", "usefulness"]
    averages = {
        c: (sum(e[c] for e in entries) / len(entries) if entries else 0) for c in criteria
    }
    return {"count": len(entries), "averages": averages, "entries": entries}


app.include_router(api)

# Sirve el build de producción del frontend (frontend/dist, generado por `npm run build`)
# desde el mismo servicio/dominio. Si no existe (p. ej. en desarrollo local, donde el
# frontend corre aparte con `npm run dev`), se omite sin error.
_FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
