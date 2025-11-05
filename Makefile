.PHONY: test docker-test

test:
	pytest services/ -v

docker-test:
	docker compose exec detector pytest -v
	docker compose exec predictor pytest -v
	docker compose exec rca pytest -v
	docker compose exec simulator pytest -v
