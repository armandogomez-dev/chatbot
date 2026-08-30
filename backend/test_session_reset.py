import asyncio
import json
from unittest.mock import patch

import pytest
from fastapi import BackgroundTasks, HTTPException
from starlette.requests import Request

from main import SUPPORT_AGENT_REMINDER, chat, chat_histories
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
            patch("main.inference.classify_sentiment", return_value=("neutral", 0.5)),
            patch(
                "main.inference.generate",
                side_effect=lambda prompt, is_risk, sentiment_label: captured_prompts.append(prompt) or "ok",
            ),
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


def test_first_turn_includes_support_agent_reminder():
    chat_histories.clear()

    async def run_flow():
        request = Request({"type": "http", "method": "POST", "path": "/chat", "headers": []})

        with (
            patch("main.inference._translate_to_en", side_effect=lambda text: text),
            patch("main.inference._translate_to_es", side_effect=lambda text: text),
            patch("main.inference.classify", return_value=("no riesgo", 0.1)),
            patch("main.inference.classify_sentiment", return_value=("neutral", 0.5)),
            patch("main.inference.generate", return_value="Hola, estoy aquí para apoyarte."),
        ):
            first_response = await chat(
                request,
                ChatRequest(message="Hola", session_id="s1", alert_sent=False),
                BackgroundTasks(),
            )
            second_response = await chat(
                request,
                ChatRequest(message="Hoy estoy mejor", session_id="s1", alert_sent=False),
                BackgroundTasks(),
            )

        return json.loads(first_response.body), json.loads(second_response.body)

    first_body, second_body = asyncio.run(run_flow())

    assert SUPPORT_AGENT_REMINDER in first_body["response"]
    assert SUPPORT_AGENT_REMINDER not in second_body["response"]


def test_professional_request_triggers_alert_immediately():
    chat_histories.clear()

    async def run_flow():
        request = Request({"type": "http", "method": "POST", "path": "/chat", "headers": []})
        background_tasks = BackgroundTasks()

        with (
            patch("main.inference._translate_to_en", side_effect=lambda text: text),
            patch("main.inference._translate_to_es", side_effect=lambda text: text),
            patch("main.inference.classify", return_value=("no riesgo", 0.1)),
            patch("main.inference.classify_sentiment", return_value=("neutral", 0.5)),
            patch("main.inference.generate", return_value="Entiendo, voy a ayudarte con eso."),
        ):
            response = await chat(
                request,
                ChatRequest(
                    message="Necesito hablar con un profesional",
                    session_id="s2",
                    alert_sent=False,
                ),
                background_tasks,
            )

        return json.loads(response.body), background_tasks

    body, background_tasks = asyncio.run(run_flow())

    assert body["alert_sent"] is True
    assert len(background_tasks.tasks) == 2


def test_alerted_session_rejects_later_messages():
    chat_histories.clear()

    async def run_flow():
        request = Request({"type": "http", "method": "POST", "path": "/chat", "headers": []})
        background_tasks = BackgroundTasks()

        with (
            patch("main.inference._translate_to_en", side_effect=lambda text: text),
            patch("main.inference._translate_to_es", side_effect=lambda text: text),
            patch("main.inference.classify", return_value=("no riesgo", 0.1)),
            patch("main.inference.classify_sentiment", return_value=("neutral", 0.5)),
            patch("main.inference.generate", return_value="Entiendo, voy a ayudarte con eso."),
        ):
            await chat(
                request,
                ChatRequest(
                    message="Necesito hablar con un profesional",
                    session_id="s3",
                    alert_sent=False,
                ),
                background_tasks,
            )

            with pytest.raises(HTTPException, match="remitido a un profesional"):
                await chat(
                    request,
                    ChatRequest(
                        message="Quiero seguir hablando",
                        session_id="s3",
                        alert_sent=False,
                    ),
                    background_tasks,
                )

    asyncio.run(run_flow())
