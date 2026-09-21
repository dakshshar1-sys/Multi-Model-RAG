# Developer shortcuts. `make help` lists them.
#
# Two facts about this stack that are easy to forget, and that these targets encode:
#   - the backend container bind-mounts ./backend but does not hot-reload: code changes need a restart;
#   - the frontend image bakes the build in, and its anonymous volumes keep the OLD .next and
#     node_modules alive across `up`: a rebuild needs --renew-anon-volumes or you keep seeing stale code.

BACKEND  := multimodelrag-backend-1
COMPOSE  := docker compose

.PHONY: help up down restart-backend rebuild-frontend logs test test-ci ci-requirements eval latency abstention

help:            ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

up:              ## start the whole stack
	$(COMPOSE) up -d

down:            ## stop the whole stack
	$(COMPOSE) down

restart-backend: ## pick up backend code changes (no hot reload), wait until healthy
	$(COMPOSE) restart backend
	@i=0; until curl -sf http://localhost:8000/api/health >/dev/null 2>&1; do i=$$((i+1)); [ $$i -ge 90 ] && { echo "backend did not become healthy"; exit 1; }; sleep 2; done; echo "backend healthy"

rebuild-frontend: ## rebuild the frontend image and drop its stale anonymous volumes
	$(COMPOSE) build frontend
	$(COMPOSE) up -d --force-recreate --renew-anon-volumes frontend

logs:            ## follow backend logs
	docker logs -f --tail 100 $(BACKEND)

test:            ## full test suite in the backend container (ML runtime present: nothing skipped)
	docker exec -w /app $(BACKEND) python -m pytest tests/ -q -rs

test-ci:         ## exactly what CI runs: clean Python 3.11, requirements-ci.txt, fresh export, no ML runtime
	@docker rm -f mmrag-ci >/dev/null 2>&1 || true
	docker run -d --name mmrag-ci python:3.11-slim sleep infinity >/dev/null
	docker exec mmrag-ci mkdir /ci
	git archive HEAD backend | docker exec -i mmrag-ci tar -x -C /ci
	docker exec -w /ci/backend mmrag-ci sh -c 'python scripts/make_ci_requirements.py --check && pip install -q --disable-pip-version-check -r requirements-ci.txt 2>&1 | grep -v "WARNING: Running pip" ; python -m pytest tests/ -q -rs'; rc=$$?; docker rm -f mmrag-ci >/dev/null; exit $$rc

ci-requirements: ## regenerate backend/requirements-ci.txt after editing requirements.txt
	python3 backend/scripts/make_ci_requirements.py

eval:            ## reproduce every number in backend/eval/BASELINE.md (~50 min)
	docker exec -w /app $(BACKEND) sh eval/run_all.sh

latency:         ## regenerate backend/eval/LATENCY_REPORT.md from recorded traces
	docker exec -w /app $(BACKEND) python -B -m eval.latency_report

abstention:      ## run the abstention evaluation (~6 min)
	docker exec -w /app $(BACKEND) python -B -W ignore -m eval.abstention
