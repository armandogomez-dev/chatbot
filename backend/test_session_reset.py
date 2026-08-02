import asyncio
from unittest.mock import patch

from fastapi import BackgroundTasks
from starlette.requests import Request

from main import chat, chat_histories
from schemas import ChatRequest


def test_new_session_clears_previous_history():
    chat_histories.clear()
    captured_prompts = []

    async def run_flow():
        request = Request({"type": "http", "method": "POST", "path": "/chat", "headers": []})

        with (
            patch("main.inference._translate_to_en", side_effect=lambda text: text),
            patch("main.inference._translate_to_es", side_effect=lambda text: text),
            patch("main.inference.classify", return_value=("normal", 0.1)),
            patch("main.inference.generate", side_effect=lambda prompt, is_risk: captured_prompts.append(prompt) or "ok"),
        ):
            await chat(
                request,
                ChatRequest(message="Hola", session_id="s1", alert_sent=False),
                BackgroundTasks(),
            )
            await chat(
                request,
                ChatRequest(message="Adiós", session_id="s1", alert_sent=False, new_session=True),
                BackgroundTasks(),
            )

    asyncio.run(run_flow())

    assert "Hola" not in captured_prompts[1]
    assert "Adiós" in captured_prompts[1]
