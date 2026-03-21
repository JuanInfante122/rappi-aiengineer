.PHONY: setup run load-data test

setup:
	docker compose build

run:
	docker compose up

load-data:
	docker compose run --rm app python data/load_data.py

test:
	docker compose exec app pytest tests/
