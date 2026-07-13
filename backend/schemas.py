from pydantic import BaseModel


class UserInfo(BaseModel):
    name: str = ""
    phone: str = ""
    email: str = ""


class RiskEntry(BaseModel):
    risk_label: str
    risk_confidence: float


class ChatRequest(BaseModel):
    message: str
    user_info: UserInfo = UserInfo()
    history: list[RiskEntry] = []
    alert_sent: bool = False


class ChatResponse(BaseModel):
    response: str
    risk_label: str
    risk_confidence: float
    alert_sent: bool
