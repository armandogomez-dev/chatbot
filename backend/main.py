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

# Umbral de confianza para disparar alertas (0.0–1.0)
RISK_ALERT_THRESHOLD = float(os.getenv("RISK_ALERT_THRESHOLD", "0.75"))


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

    if risk_label == "riesgo" and risk_confidence >= RISK_ALERT_THRESHOLD:
        ui = body.user_info
        background_tasks.add_task(send_email_alert, text, risk_confidence, ui.name, ui.phone, ui.email)
        background_tasks.add_task(send_whatsapp_alert, text, risk_confidence, ui.name, ui.phone, ui.email)

    return ChatResponse(
        response=response,
        risk_label=risk_label,
        risk_confidence=risk_confidence,
    )
