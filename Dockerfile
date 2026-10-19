# ---- stage 1: build the React frontend -------------------------------------------------
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- stage 2: API + analytics engine + built frontend ------------------------------------
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 APP_ENV=prod RECOVER_STALE_ON_START=0
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
 && useradd -m -u 10001 app
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY --chown=app:app . .
COPY --from=web --chown=app:app /web/dist ./frontend/dist
RUN mkdir -p /app/data && chown app:app /app/data
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/api/health', timeout=4)" || exit 1
CMD ["sh", "-c", "python -m backend.app.prestart && exec uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers ${WEB_WORKERS:-2}"]
