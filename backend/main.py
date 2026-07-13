import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from inference import inference
from notifier import send_email_alert, send_whatsapp_alert
from schemas import ChatRequest, ChatResponse

logging.basicConfig(level=logging.INFO)

# Umbral de confianza para considerar un mensaje como "alto riesgo" (0.0–1.0)
RISK_ALERT_THRESHOLD = float(os.getenv("RISK_ALERT_THRESHOLD", "0.75"))

# Se dispara la alerta al especialista cuando, de los últimos RISK_ALERT_WINDOW
# mensajes (incluyendo el actual), al menos RISK_ALERT_MIN_COUNT son de alto riesgo.
RISK_ALERT_WINDOW = int(os.getenv("RISK_ALERT_WINDOW", "5"))
RISK_ALERT_MIN_COUNT = int(os.getenv("RISK_ALERT_MIN_COUNT", "2"))


def _is_high_risk(label: str, confidence: float) -> bool:
    return label == "riesgo" and confidence >= RISK_ALERT_THRESHOLD


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


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, background_tasks: BackgroundTasks):
    text = body.message.strip()
    if not text:
        raise HTTPException(status_code=422, detail="El mensaje no puede estar vacío.")

    response, risk_label, risk_confidence = inference.chat(text)

    recent = [(e.risk_label, e.risk_confidence) for e in body.history[-(RISK_ALERT_WINDOW - 1):]]
    recent.append((risk_label, risk_confidence))
    high_risk_count = sum(1 for label, confidence in recent if _is_high_risk(label, confidence))

    alert_sent = body.alert_sent
    if not body.alert_sent and high_risk_count >= RISK_ALERT_MIN_COUNT:
        ui = body.user_info
        background_tasks.add_task(send_email_alert, text, risk_confidence, ui.name, ui.phone, ui.email)
        background_tasks.add_task(send_whatsapp_alert, text, risk_confidence, ui.name, ui.phone, ui.email)
        alert_sent = True

    return ChatResponse(
        response=response,
        risk_label=risk_label,
        risk_confidence=risk_confidence,
        alert_sent=alert_sent,
    )
