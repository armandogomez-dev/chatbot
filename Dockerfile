# --- Stage 1: build del frontend (React + Vite) ---
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- Stage 2: backend (FastAPI + modelos) sirviendo también el build del frontend ---
FROM python:3.11-slim AS backend
WORKDIR /app

# Instala primero las ruedas CPU-only de torch (las de PyPI son la build CUDA, mucho más
# pesadas e inútiles aquí: Railway no da GPU). Las versiones deben calzar con
# backend/requirements.txt; pip no reinstala si la versión ya calza, así que el
# `pip install -r requirements.txt` de abajo las deja intactas.
RUN pip install --no-cache-dir torch==2.12.0 torchvision==0.27.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ ./backend/
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

WORKDIR /app/backend
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
