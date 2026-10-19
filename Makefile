.PHONY: install dev api web test build smoke docker
install:  ## python + node deps
	pip install -r requirements-dev.txt && cd frontend && npm ci
api:      ## backend with auto-reload on :8000
	uvicorn backend.app.main:app --reload --port 8000
web:      ## frontend dev server on :5173 (proxies /api to :8000)
	cd frontend && npm run dev
test:
	python -m pytest -q && cd frontend && npm run typecheck && npm test
build:
	cd frontend && npm run build
smoke: build
	python scripts/smoke_e2e.py
docker:
	docker compose up --build
