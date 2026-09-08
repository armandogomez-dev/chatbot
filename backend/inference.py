import os
import random
import re
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import (
    AutoTokenizer,
    MarianMTModel,
    MarianTokenizer,
    RobertaForSequenceClassification,
    T5ForConditionalGeneration,
)

BASE_DIR = Path(__file__).parent.parent

# En local, cada modelo se carga desde su carpeta del repo. En producción (Railway) los
# pesos no viven en git (son ~1.2GB) — estas env vars apuntan en su lugar a un repo de
# Hugging Face Hub (p. ej. "usuario/chatbox-modelo-salo"); si el repo es privado, además
# hace falta la env var HF_TOKEN, que transformers/huggingface_hub leen automáticamente.
MODEL_CLASSIFIER_PATH = os.getenv("MODEL_CLASSIFIER_REPO", str(BASE_DIR / "modelo salo"))
MODEL_GENERATOR_PATH = os.getenv("MODEL_GENERATOR_REPO", str(BASE_DIR / "modelo jhon"))
MODEL_SENTIMENT_PATH = os.getenv("MODEL_SENTIMENT_REPO", str(BASE_DIR / "sentiment_model"))

# Modelo de embeddings multilingüe (no se reentrena, solo se usa para medir similitud de
# significado) usado por la red de seguridad de riesgo — ver más abajo.
RISK_EMBEDDING_MODEL = os.getenv(
    "RISK_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
RISK_SIMILARITY_THRESHOLD = float(os.getenv("RISK_SIMILARITY_THRESHOLD", "0.65"))

# HuggingFace translation models (downloaded automatically on first run)
TRANS_ES_EN = os.getenv("TRANS_ES_EN_MODEL", "Helsinki-NLP/opus-mt-es-en")
TRANS_EN_ES = os.getenv("TRANS_EN_ES_MODEL", "Helsinki-NLP/opus-mt-en-es")

T5_PREFIX_RISK = os.getenv("T5_PREFIX_RISK", "riesgo:").strip()
T5_PREFIX_NORMAL = os.getenv("T5_PREFIX_NORMAL", "chat:").strip()

# Marcadores de basura que el generador a veces suelta al final de una respuesta,
# después de una o dos oraciones coherentes: markdown residual y disclaimers tipo
# "as an AI language model". Todo lo que aparezca a partir del primero de estos se recorta.
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

# El generador de riesgo a veces repite el prefijo del prompt ("riesgo:"/"chat:")
# como si fuera parte de su propia respuesta; nunca debe llegar así al usuario.
_PROMPT_PREFIX_ECHO_MARKERS = ("riesgo:", "chat:")

# Red de seguridad por similitud semántica sobre el texto en español ORIGINAL (antes de
# traducir). El clasificador ML (modelo salo) puede fallar en ideación suicida/autolesión
# pasiva o poco explícita (falsos negativos verificados en pruebas reales: "sería más fácil
# para todos si yo no estuviera" → 92% "sin riesgo"; "me siento mal con ganas de hacerme
# daño" → 95% "sin riesgo"). En vez de una lista de frases exactas (frágil: cualquier
# parafraseo nuevo se escapa), se mide qué tan cerca está el significado del mensaje del de
# estas frases ancla usando un modelo de embeddings ya entrenado (no se entrena nada acá).
# Esto generaliza a paráfrasis no vistas ("tengo ganas de hacerme daño", "pienso en
# lastimarme", "me quiero cortar" caen todas cerca en significado). Nunca degrada una
# detección de riesgo que el modelo ya hizo por su cuenta, solo corrige falsos negativos.
# Umbral calibrado a mano (ver RISK_SIMILARITY_THRESHOLD): con el modelo mpnet-base-v2,
# los casos de riesgo verificados dieron 0.63–0.95 de similitud y los mensajes benignos de
# control se quedaron en ≤0.62, con margen hasta 0.65.
_RISK_ANCHOR_SENTENCES = (
    "A veces pienso que sería mejor si yo no estuviera aquí.",
    "Siento que ya no vale la pena seguir viviendo.",
    "No le encuentro sentido a seguir con mi vida.",
    "Tengo ganas de hacerme daño.",
    "Pienso en lastimarme a mí mismo.",
    "Quiero terminar con todo de una vez.",
    "A veces pienso en quitarme la vida.",
    "Siento que soy una carga para todos los que me rodean.",
    "Ya no aguanto más esta situación, quiero desaparecer.",
    "Nadie notaría si dejo de existir.",
    "He pensado en maneras de hacerme daño.",
    "Quiero morirme y que todo esto acabe.",
    "No tengo ganas de vivir un día más.",
    "Estoy pensando en suicidarme.",
    "Me quiero cortar para sentir algo diferente.",
    "Siento que todos estarían mejor sin mí.",
    "No veo ninguna salida a lo que estoy viviendo.",
    "A veces fantaseo con no despertar mañana.",
    "Ya no quiero seguir luchando, quiero rendirme por completo.",
    "Pienso en hacerme daño cuando me siento así de mal.",
)


class ChatInference:
    def __init__(self) -> None:
        self._classifier: RobertaForSequenceClassification | None = None
        self._clf_tokenizer = None
        self._generator: T5ForConditionalGeneration | None = None
        self._gen_tokenizer = None
        self._sentiment_classifier: RobertaForSequenceClassification | None = None
        self._sentiment_tokenizer = None
        self._marian_es_en: MarianMTModel | None = None
        self._marian_es_en_tok: MarianTokenizer | None = None
        self._marian_en_es: MarianMTModel | None = None
        self._marian_en_es_tok: MarianTokenizer | None = None
        self._risk_embedder: SentenceTransformer | None = None
        self._risk_anchor_embeddings = None
        self.default_farewell_messages = (
            "Estoy aquí para servirte siempre que lo necesites.",
            "Fue un gusto acompañarte. Vuelve cuando quieras hablar.",
            "Aquí estaré cuando lo necesites. Cuídate mucho.",
        )
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

        print("Cargando modelo de embeddings para red de seguridad de riesgo...")
        self._risk_embedder = SentenceTransformer(RISK_EMBEDDING_MODEL)
        self._risk_anchor_embeddings = self._risk_embedder.encode(
            list(_RISK_ANCHOR_SENTENCES), convert_to_tensor=True, normalize_embeddings=True
        )
        print("Modelos listos.")

    def _translate(self, text: str, model: MarianMTModel, tokenizer: MarianTokenizer) -> str:
        inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            # num_beams=1 (greedy): en CPU, la búsqueda por beams del checkpoint opus-mt
            # (num_beams=4/6 por defecto) es uno de los pasos más caros del pipeline;
            # para oraciones cortas la pérdida de fluidez frente a beam search es mínima.
            output_ids = model.generate(**inputs, num_beams=1, max_length=200)
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

    def _max_risk_similarity(self, text_es: str) -> float:
        """Similitud coseno (0–1) entre el significado del mensaje y la frase ancla de
        riesgo más parecida. No es una probabilidad calibrada, solo una medida de cercanía
        semántica — de ahí el umbral calibrado a mano (RISK_SIMILARITY_THRESHOLD)."""
        embedding = self._risk_embedder.encode(
            text_es, convert_to_tensor=True, normalize_embeddings=True
        )
        similarities = util.cos_sim(embedding, self._risk_anchor_embeddings)
        return float(similarities.max().item())

    def apply_risk_safety_net(self, risk_label: str, confidence: float, text_es: str) -> tuple[str, float]:
        """Red de seguridad por similitud semántica sobre el mensaje en español original: si
        el clasificador ML dice 'no riesgo' pero el texto se parece en significado a una frase
        ancla de riesgo conocida, se fuerza 'riesgo'. La confianza asignada siempre queda por
        encima de RISK_ALERT_THRESHOLD (0.75 por defecto) para que SÍ cuente en la alerta por
        acumulación — si esta red decide que el mensaje es de riesgo, debe pesar como tal.
        Nunca degrada una detección de riesgo que el modelo ya hizo por su cuenta."""
        if risk_label == "riesgo":
            return risk_label, confidence
        similarity = self._max_risk_similarity(text_es)
        if similarity >= RISK_SIMILARITY_THRESHOLD:
            forced_confidence = min(0.98, max(0.80, similarity + 0.15))
            return "riesgo", forced_confidence
        return risk_label, confidence

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

        cleaned_lower = cleaned.lower()
        for marker in _PROMPT_PREFIX_ECHO_MARKERS:
            if cleaned_lower.startswith(marker):
                cleaned = cleaned[len(marker):].lstrip()
                break

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
                # max_length/num_beams bajados para CPU: con 4 beams y 200 tokens este era
                # el paso más lento del pipeline (el costo crece ~linealmente con num_beams).
                max_length=120,
                num_beams=2,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.3,
            )
        raw = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return self._strip_generation_artifacts(raw)

    def generate(self, text_en: str, is_risk: bool, sentiment_label: str) -> str:
        # El checkpoint que antes atendía sentimiento neutral/positivo (Chatbot_converncional_v1)
        # divergió durante su entrenamiento (loss de validación se dispara y nunca se recupera
        # a partir del step ~8500) y producía texto degenerado. Se retiró: todo mensaje sin
        # riesgo pasa ahora por "modelo jhon" con el prefijo "chat:", que en pruebas reales
        # da respuestas coherentes para cualquier tono (sentiment_label queda sin usar aquí,
        # se conserva en la firma para no tocar a quien la llama).
        del sentiment_label
        prefix = T5_PREFIX_RISK if is_risk else T5_PREFIX_NORMAL
        response = self._generate_with(self._generator, self._gen_tokenizer, f"{prefix} {text_en}")

        if self._is_degenerate_generation(response):
            response = self.default_generation_fallback_en
        return response

    def _normalize_for_farewell_detection(self, text: str) -> str:
        normalized = re.sub(r"[^a-záéíóúüñ\s]", " ", text.lower())
        return " ".join(normalized.split())

    # "buenas tardes/noches/días" quedan fuera a propósito: en español se usan tanto para
    # saludar como para despedirse, y como saludo son casi siempre el primer mensaje de la
    # conversación. "gracias" solo cuenta como despedida cuando lo dice el usuario: el
    # generador lo usa todo el tiempo como apertura empática ("gracias por compartir eso"),
    # así que en la respuesta del bot ese marcador daría falsos positivos constantemente.
    _UNAMBIGUOUS_FAREWELL_MARKERS = [
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
        "cuídate",
        "cuidate",
    ]

    def should_add_default_closing(self, text: str, include_gracias: bool = False) -> bool:
        normalized = self._normalize_for_farewell_detection(text)
        markers = self._UNAMBIGUOUS_FAREWELL_MARKERS
        if include_gracias:
            markers = markers + ["gracias"]
        return any(marker in normalized for marker in markers)

    def append_default_closing(self, response: str, user_text: str | None = None) -> str:
        if user_text and self.should_add_default_closing(user_text, include_gracias=True):
            return random.choice(self.default_farewell_messages)

        if self.should_add_default_closing(response):
            return random.choice(self.default_farewell_messages)
        return response

    def chat(self, text_es: str) -> tuple[str, str, float]:
        """Full pipeline: Spanish in, Spanish out. Uses only the current message —
        the generators are fine-tuned on single-turn inputs, not conversation history.
        """
        # Translate current message to English for risk/sentiment classification (current message only)
        text_en = self._translate_to_en(text_es)
        risk_label, confidence = self.classify(text_en)
        risk_label, confidence = self.apply_risk_safety_net(risk_label, confidence, text_es)
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
