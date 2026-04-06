# Simple helper targets for willitrun packaging and data pipeline

PYTHON ?= python

.PHONY: fetch build_db status wheel all clean check_db_sync

fetch:
	$(PYTHON) scripts/run_pipeline.py fetch

build_db:
	$(PYTHON) scripts/run_pipeline.py build

status:
	$(PYTHON) scripts/run_pipeline.py status

check_db_sync:
	@[ -f data/benchmarks.db ] || (echo "data/benchmarks.db missing; run make build_db" && exit 1)
	@[ -f willitrun/data/benchmarks.db ] || (echo "willitrun/data/benchmarks.db missing; run make wheel" && exit 1)
	cmp data/benchmarks.db willitrun/data/benchmarks.db >/dev/null || (echo "Benchmark DBs differ; rebuild and copy." && exit 1)

wheel: build_db
	cp data/benchmarks.db willitrun/data/benchmarks.db
	[ -f data/benchmarks.meta.json ] && cp data/benchmarks.meta.json willitrun/data/benchmarks.meta.json || true
	$(PYTHON) -m build

all: fetch build_db

clean:
	rm -f data/benchmarks.db willitrun/data/benchmarks.db data/benchmarks.meta.json willitrun/data/benchmarks.meta.json
	rm -rf dist build *.egg-info
