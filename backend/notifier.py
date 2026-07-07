import logging
import os
import smtplib
from datetime import datetime
from email.message import EmailMessage

logger = logging.getLogger(__name__)

PROFESSIONAL_EMAIL = os.getenv("PROFESSIONAL_EMAIL", "")
PROFESSIONAL_PHONE = os.getenv("PROFESSIONAL_PHONE", "")  # E.164 colombiano: +573001234567

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")  # App Password de Gmail

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
# Sandbox Twilio: +14155238886 | En producción reemplazar con número propio
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")


def _user_contact_block(user_name: str, user_phone: str, user_email: str) -> str:
    lines = []
    if user_name:
        lines.append(f"Nombre: {user_name}")
    if user_phone:
        lines.append(f"Teléfono/WhatsApp: {user_phone}")
    if user_email:
        lines.append(f"Correo: {user_email}")
    if not lines:
        return "Datos de contacto: no proporcionados"
    return "\n".join(lines)


def _alert_body(
    user_message: str,
    risk_confidence: float,
    user_name: str = "",
    user_phone: str = "",
    user_email: str = "",
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    contact = _user_contact_block(user_name, user_phone, user_email)
    return (
        f"ALERTA DE RIESGO ALTO - Chatbox de Apoyo\n"
        f"Hora: {ts}\n"
        f"Nivel de confianza: {risk_confidence:.1%}\n\n"
        f"--- Datos del usuario ---\n"
        f"{contact}\n\n"
        f"--- Mensaje del usuario ---\n"
        f"{user_message}\n\n"
        f"Por favor comuníquese con el usuario a la brevedad posible."
    )


def send_email_alert(
    user_message: str,
    risk_confidence: float,
    user_name: str = "",
    user_phone: str = "",
    user_email: str = "",
) -> None:
    if not all([PROFESSIONAL_EMAIL, SMTP_USER, SMTP_PASSWORD]):
        logger.warning("Alerta por correo omitida: SMTP no configurado (revisar variables de entorno).")
        return
    try:
        name_tag = f" · {user_name}" if user_name else ""
        msg = EmailMessage()
        msg["Subject"] = f"ALERTA: Usuario en posible riesgo{name_tag} ({risk_confidence:.0%} confianza)"
        msg["From"] = SMTP_USER
        msg["To"] = PROFESSIONAL_EMAIL
        msg.set_content(_alert_body(user_message, risk_confidence, user_name, user_phone, user_email))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(SMTP_USER, SMTP_PASSWORD)
            smtp.send_message(msg)

        logger.info("Alerta por correo enviada a %s", PROFESSIONAL_EMAIL)
    except Exception:
        logger.exception("Error al enviar alerta por correo")


def send_whatsapp_alert(
    user_message: str,
    risk_confidence: float,
    user_name: str = "",
    user_phone: str = "",
    user_email: str = "",
) -> None:
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, PROFESSIONAL_PHONE]):
        logger.warning("Alerta por WhatsApp omitida: Twilio no configurado (revisar variables de entorno).")
        return
    try:
        from twilio.rest import Client

        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            from_=TWILIO_WHATSAPP_FROM,
            to=f"whatsapp:{PROFESSIONAL_PHONE}",
            body=_alert_body(user_message, risk_confidence, user_name, user_phone, user_email),
        )
        logger.info("Alerta por WhatsApp enviada a %s", PROFESSIONAL_PHONE)
    except Exception:
        logger.exception("Error al enviar alerta por WhatsApp")
