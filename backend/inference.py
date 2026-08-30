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

# Marcadores de basura que el checkpoint del generador conversacional (sobreajustado)
# suele soltar al final de una respuesta, después de una o dos oraciones coherentes:
# markdown residual y disclaimers tipo "as an AI language model". Todo lo que aparezca
# a partir del primero de estos se recorta.
_GENERATION_GARBAGE_MARKERS = ("###", "**", "---")
_GENERATION_DISCLAIMER_PHRASES = (
    "as an ai language model",
    "as an ai,",
    "as a language model",
    "i'm just an ai",
    "i am an ai",
    "i don't have access to",
    "i don't have a computer",
)
_MIN_VALID_GENERATION_LENGTH = 3


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
        self.default_generation_fallback_en = (
            "I'm here for you. Tell me a bit more about how you're feeling."
        )

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

    def _strip_generation_artifacts(self, text: str) -> str:
        """Corta la respuesta en el primer marcador de basura (markdown residual o
        disclaimer de IA) que suelta el checkpoint sobreajustado, y descarta una
        oración final que quede colgando en ':' (p. ej. "...today:") por el corte."""
        lower = text.lower()
        cut_at = len(text)
        for marker in _GENERATION_GARBAGE_MARKERS:
            idx = text.find(marker)
            if idx != -1:
                cut_at = min(cut_at, idx)
        for phrase in _GENERATION_DISCLAIMER_PHRASES:
            idx = lower.find(phrase)
            if idx != -1:
                cut_at = min(cut_at, idx)
        cleaned = text[:cut_at].rstrip()

        if cleaned.endswith(":"):
            sentences = re.split(r"(?<=[.!?])\s+", cleaned)
            if sentences and sentences[-1].rstrip().endswith(":"):
                sentences = sentences[:-1]
            cleaned = " ".join(sentences).strip()

        return cleaned

    def _is_degenerate_generation(self, text: str) -> bool:
        return len(text.strip()) < _MIN_VALID_GENERATION_LENGTH or not re.search(r"[A-Za-z]{2,}", text)

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
        raw = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return self._strip_generation_artifacts(raw)

    def generate(self, text_en: str, is_risk: bool, sentiment_label: str) -> str:
        # Riesgo o sentimiento negativo: modelo jhon, ya validado para estos casos.
        # Sentimiento neutral/positivo sin riesgo: Chatbot_converncional_v1, entrenado para eso
        # pero sobreajustado (ver _strip_generation_artifacts / _is_degenerate_generation).
        if is_risk or sentiment_label == "negative":
            prefix = T5_PREFIX_RISK if is_risk else T5_PREFIX_NORMAL
            response = self._generate_with(self._generator, self._gen_tokenizer, f"{prefix} {text_en}")
        else:
            response = self._generate_with(self._generator_positive, self._gen_positive_tokenizer, text_en)

        if self._is_degenerate_generation(response):
            response = self.default_generation_fallback_en
        return response

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

    def chat(self, text_es: str) -> tuple[str, str, float]:
        """Full pipeline: Spanish in, Spanish out. Uses only the current message —
        the generators are fine-tuned on single-turn inputs, not conversation history.
        """
        # Translate current message to English for risk/sentiment classification (current message only)
        text_en = self._translate_to_en(text_es)
        risk_label, confidence = self.classify(text_en)
        is_risk = risk_label == "riesgo"
        sentiment_label, _ = self.classify_sentiment(text_en)

        # Generate response in English from the current message only. The generators were
        # fine-tuned on single-turn inputs ("riesgo: <msg>" / "chat: <msg>"), not multi-turn
        # "Usuario:/Asistente:" transcripts, so feeding them the full history makes them echo
        # the prompt back instead of producing a new reply.
        response_en = self.generate(text_en, is_risk, sentiment_label)

        # Translate response to Spanish
        response_es = self._translate_to_es(response_en)
        response_es = self.append_default_closing(response_es, text_es)

        return response_es, risk_label, confidence


inference = ChatInference()
