import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference import inference
from notifier import send_email_alert, send_whatsapp_alert
from schemas import ChatRequest, ChatResponse, UserInfo, RiskEntry

logging.basicConfig(level=logging.INFO)

# Umbral de confianza para considerar un mensaje como "alto riesgo" (0.0–1.0)
RISK_ALERT_THRESHOLD = float(os.getenv("RISK_ALERT_THRESHOLD", "0.75"))

# Se dispara la alerta al especialista cuando, de los últimos RISK_ALERT_WINDOW
# mensajes (incluyendo el actual), al menos RISK_ALERT_MIN_COUNT son de alto riesgo.
RISK_ALERT_WINDOW = int(os.getenv("RISK_ALERT_WINDOW", "5"))
RISK_ALERT_MIN_COUNT = int(os.getenv("RISK_ALERT_MIN_COUNT", "2"))


def _is_high_risk(label: str, confidence: float) -> bool:
    return label == "riesgo" and confidence >= RISK_ALERT_THRESHOLD


# In-memory store for chat histories: session_id -> list of messages
# Each message is a dict: {"role": "user" or "assistant", "content": str, "timestamp": float}
# We'll keep it simple and just store the last N messages (where N is RISK_ALERT_WINDOW * 2?).
# But we want to keep the entire conversation for context? Let's store the last 20 messages.
chat_histories: Dict[str, List[dict]] = {}
MAX_HISTORY_LENGTH = 20  # keep last 20 messages (10 turns)


@asynccontextmanager
async def lifespan(app: FastAPI):
    inference.load()
    yield


app = FastAPI(title="Chatbox API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
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
        session_id = body.session_id or str(uuid4())
    else:
        session_id = body.session_id or request.cookies.get("session_id")
        if session_id is None:
            session_id = str(uuid4())

    # Retrieve history for this session (list of messages with role and content)
    history_raw = chat_histories.get(session_id, [])

    # Compute risk for the current user message (needed for storage and alert)
    text_en = inference._translate_to_en(text)
    risk_label, risk_confidence = inference.classify(text_en)
    is_risk = risk_label == "riesgo"

    # Build conversation history string for generation (using only role and content from history_raw)
    history_str = ""
    for msg in history_raw:
        role = msg['role']
        content = msg['content']
        if role == 'user':
            history_str += f"Usuario: {content}\n"
        else:  # 'assistant'
            history_str += f"Asistente: {content}\n"
    history_str += f"Usuario: {text}\nAsistente: "

    # Translate the entire history string to English for the generator
    try:
        prompt_en = inference._translate_to_en(history_str)
    except Exception as e:
        # Fallback to translating only the current message if history translation fails
        print(f"Translation error for history: {e}. Falling back to current message only.")
        prompt_en = text_en

    # Generate response in English
    response_en = inference.generate(prompt_en, is_risk)
    response_es = inference._translate_to_es(response_en)

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

    alert_sent = body.alert_sent
    if not body.alert_sent and high_risk_count >= RISK_ALERT_MIN_COUNT:
        print(f"ALERT TRIGGERED: high_risk_count={high_risk_count}, threshold={RISK_ALERT_MIN_COUNT}")
        ui = body.user_info
        background_tasks.add_task(send_email_alert, text, risk_confidence, ui.name, ui.phone, ui.email)
        background_tasks.add_task(send_whatsapp_alert, text, risk_confidence, ui.name, ui.phone, ui.email)
        alert_sent = True
    else:
        print(f"ALERT CHECK: high_risk_count={high_risk_count}, threshold={RISK_ALERT_MIN_COUNT}, alert_sent={alert_sent}")

    # Prepare JSON response and set cookie
    response_content = {
        "response": response_es,
        "risk_label": risk_label,
        "risk_confidence": risk_confidence,
        "alert_sent": alert_sent,
        "session_id": session_id
    }
    response = JSONResponse(content=response_content)
    # Set cookie (httpOnly=False so JS can read if needed; adjust as needed)
    response.set_cookie(key="session_id", value=session_id, httponly=False, max_age=60*60*24*30)  # 30 days
    return response
