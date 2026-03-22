.PHONY: setup run dev down test test-local load-data logs

# --- Docker targets ---

setup:
	docker compose build

run:
	docker compose up

down:
	docker compose down

logs:
	docker compose logs -f

load-data:
	docker compose run --rm app python data/load_data.py

test:
	docker compose run --rm app python -m pytest tests/ -v

# --- Local targets (no Docker) ---

dev:
	streamlit run app.py

test-local:
	python -m pytest tests/ -v
