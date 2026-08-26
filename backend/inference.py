import os
import re
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    MarianMTModel,
    MarianTokenizer,
    RobertaForSequenceClassification,
    T5ForConditionalGeneration,
)

BASE_DIR = Path(__file__).parent.parent
MODEL_CLASSIFIER_PATH = str(BASE_DIR / "modelo salo")
MODEL_GENERATOR_PATH = str(BASE_DIR / "modelo jhon")
MODEL_SENTIMENT_PATH = str(BASE_DIR / "sentiment_model")
MODEL_GENERATOR_POSITIVE_PATH = str(BASE_DIR / "Chatbot_converncional_v1" / "checkpoint-8000")

# HuggingFace translation models (downloaded automatically on first run)
TRANS_ES_EN = os.getenv("TRANS_ES_EN_MODEL", "Helsinki-NLP/opus-mt-es-en")
TRANS_EN_ES = os.getenv("TRANS_EN_ES_MODEL", "Helsinki-NLP/opus-mt-en-es")

T5_PREFIX_RISK = os.getenv("T5_PREFIX_RISK", "riesgo:").strip()
T5_PREFIX_NORMAL = os.getenv("T5_PREFIX_NORMAL", "chat:").strip()


class ChatInference:
    def __init__(self) -> None:
        self._classifier: RobertaForSequenceClassification | None = None
        self._clf_tokenizer = None
        self._generator: T5ForConditionalGeneration | None = None
        self._gen_tokenizer = None
        self._sentiment_classifier: RobertaForSequenceClassification | None = None
        self._sentiment_tokenizer = None
        self._generator_positive: T5ForConditionalGeneration | None = None
        self._gen_positive_tokenizer = None
        self._marian_es_en: MarianMTModel | None = None
        self._marian_es_en_tok: MarianTokenizer | None = None
        self._marian_en_es: MarianMTModel | None = None
        self._marian_en_es_tok: MarianTokenizer | None = None
        self.default_farewell_message = "Estoy aquí para servirte siempre que lo necesites."

    def load(self) -> None:
        print("Cargando modelos de traducción...")
        self._marian_es_en_tok = MarianTokenizer.from_pretrained(TRANS_ES_EN)
        self._marian_es_en = MarianMTModel.from_pretrained(TRANS_ES_EN)
        self._marian_es_en.eval()

        self._marian_en_es_tok = MarianTokenizer.from_pretrained(TRANS_EN_ES)
        self._marian_en_es = MarianMTModel.from_pretrained(TRANS_EN_ES)
        self._marian_en_es.eval()

        print("Cargando clasificador de riesgo (modelo salo)...")
        self._clf_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CLASSIFIER_PATH, use_fast=True
        )
        self._classifier = RobertaForSequenceClassification.from_pretrained(
            MODEL_CLASSIFIER_PATH
        )
        self._classifier.eval()

        print("Cargando generador (modelo jhon)...")
        self._gen_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_GENERATOR_PATH, use_fast=True
        )
        self._generator = T5ForConditionalGeneration.from_pretrained(
            MODEL_GENERATOR_PATH
        )
        self._generator.eval()

        print("Cargando clasificador de sentimiento (sentiment_model)...")
        self._sentiment_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_SENTIMENT_PATH, use_fast=True
        )
        self._sentiment_classifier = RobertaForSequenceClassification.from_pretrained(
            MODEL_SENTIMENT_PATH
        )
        self._sentiment_classifier.eval()

        print("Cargando generador para conversación neutral/positiva (Chatbot_converncional_v1)...")
        self._gen_positive_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_GENERATOR_POSITIVE_PATH, use_fast=True
        )
        self._generator_positive = T5ForConditionalGeneration.from_pretrained(
            MODEL_GENERATOR_POSITIVE_PATH
        )
        self._generator_positive.eval()
        print("Modelos listos.")

    def _translate(self, text: str, model: MarianMTModel, tokenizer: MarianTokenizer) -> str:
        inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            output_ids = model.generate(**inputs)
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def _translate_to_en(self, text: str) -> str:
        return self._translate(text, self._marian_es_en, self._marian_es_en_tok)

    def _translate_to_es(self, text: str) -> str:
        return self._translate(text, self._marian_en_es, self._marian_en_es_tok)

    def classify(self, text_en: str) -> tuple[str, float]:
        inputs = self._clf_tokenizer(
            text_en, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            logits = self._classifier(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        idx = int(logits.argmax().item())
        label: str = self._classifier.config.id2label[idx]
        confidence: float = probs[0][idx].item()
        return label, confidence

    def classify_sentiment(self, text_en: str) -> tuple[str, float]:
        inputs = self._sentiment_tokenizer(
            text_en, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            logits = self._sentiment_classifier(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        idx = int(logits.argmax().item())
        label: str = self._sentiment_classifier.config.id2label[idx]
        confidence: float = probs[0][idx].item()
        return label, confidence

    def _generate_with(self, model: T5ForConditionalGeneration, tokenizer, text_en: str) -> str:
        inputs = tokenizer(text_en, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=200,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def generate(self, text_en: str, is_risk: bool, sentiment_label: str) -> str:
        # Riesgo o sentimiento negativo: modelo jhon, ya validado para estos casos.
        # Sentimiento neutral/positivo sin riesgo: Chatbot_converncional_v1, entrenado para eso.
        if is_risk or sentiment_label == "negative":
            prefix = T5_PREFIX_RISK if is_risk else T5_PREFIX_NORMAL
            return self._generate_with(self._generator, self._gen_tokenizer, f"{prefix} {text_en}")
        return self._generate_with(self._generator_positive, self._gen_positive_tokenizer, text_en)

    def _normalize_for_farewell_detection(self, text: str) -> str:
        normalized = re.sub(r"[^a-záéíóúüñ\s]", " ", text.lower())
        return " ".join(normalized.split())

    def should_add_default_closing(self, text: str) -> bool:
        normalized = self._normalize_for_farewell_detection(text)
        farewell_markers = [
            "gracias",
            "adios",
            "adiós",
            "chau",
            "chao",
            "hasta luego",
            "hasta la vista",
            "hasta pronto",
            "nos vemos",
            "bye",
            "desped",
            "me voy",
            "me retiro",
            "ya nos vemos",
            "que te vaya",
            "que te vaya bien",
            "buenas noches",
            "buenos días",
            "buen dia",
            "buenas tardes",
            "cuídate",
            "cuidate",
        ]
        return any(marker in normalized for marker in farewell_markers)

    def append_default_closing(self, response: str, user_text: str | None = None) -> str:
        if user_text and self.should_add_default_closing(user_text):
            return self.default_farewell_message

        if self.should_add_default_closing(response):
            return self.default_farewell_message
        return response

    def chat(self, text_es: str, history: list[dict] | None = None) -> tuple[str, str, float]:
        """Full pipeline: Spanish in, Spanish out.
        If history is provided, it is used to build a conversational prompt for the generator.
        History is a list of dicts with keys 'role' and 'content' (in Spanish), representing
        the conversation so far (excluding the current message).
        """
        # Translate current message to English for risk/sentiment classification (current message only)
        text_en = self._translate_to_en(text_es)
        risk_label, confidence = self.classify(text_en)
        is_risk = risk_label == "riesgo"
        sentiment_label, _ = self.classify_sentiment(text_en)

        # Build conversation string in Spanish for generation
        history_str = ""
        if history:
            for msg in history:
                role = msg['role']
                content = msg['content']
                if role == 'user':
                    history_str += f"Usuario: {content}\n"
                else:  # 'assistant'
                    history_str += f"Asistente: {content}\n"
        history_str += f"Usuario: {text_es}\nAsistente: "

        # Translate the entire history string to English for the generator
        prompt_en = self._translate_to_en(history_str)

        # Generate response in English
        response_en = self.generate(prompt_en, is_risk, sentiment_label)

        # Translate response to Spanish
        response_es = self._translate_to_es(response_en)
        response_es = self.append_default_closing(response_es, text_es)

        return response_es, risk_label, confidence


inference = ChatInference()
