.PHONY: dev test frontend-build up up-ascend down logs

dev:
	cd backend && uvicorn app.main:app --reload --port 8000

test:
	cd backend && python -m pytest -q

frontend-build:
	cd frontend && npm install && npm run build

up:
	docker compose -f compose.yml up -d --build

up-ascend:
	docker compose -f compose.yml -f compose.ascend.yml up -d --build

down:
	docker compose -f compose.yml -f compose.ascend.yml down

logs:
	docker compose logs -f --tail=200 app

