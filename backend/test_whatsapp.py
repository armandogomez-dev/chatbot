from dotenv import load_dotenv
load_dotenv()

from notifier import send_whatsapp_alert

send_whatsapp_alert(
    user_message="Estoy pensando en hacerme daño, no veo salida.",
    risk_confidence=0.92,
    user_name="Prueba Sistema",
    user_phone="+573000000000",
    user_email="prueba@test.com",
)

print("Mensaje enviado — revisa tu WhatsApp.")
