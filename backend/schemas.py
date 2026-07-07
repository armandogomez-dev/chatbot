from pydantic import BaseModel


class UserInfo(BaseModel):
    name: str = ""
    phone: str = ""
    email: str = ""


class ChatRequest(BaseModel):
    message: str
    user_info: UserInfo = UserInfo()


class ChatResponse(BaseModel):
    response: str
    risk_label: str
    risk_confidence: float
