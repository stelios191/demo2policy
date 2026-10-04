# Entry points. Run `make` to list them.

PYTHON ?= .venv/bin/python
UPSTREAM ?= ../upstream
ROBOMIMIC_COMMIT := d309eaecc18acf4152a830a895a6984b8ac71b05
ROBOSUITE_TAG := v1.5.1
VENV_PATH := PATH="$(CURDIR)/.venv/bin:$$PATH"

.PHONY: help setup test lint check reproduce-quick reproduce reproduce-full

help:  ## list the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-16s %s\n", $$1, $$2}'

setup:  ## create .venv with the pinned simulator stack (needs uv and git)
	@command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/"; exit 1; }
	@mkdir -p $(UPSTREAM)
	@test -d $(UPSTREAM)/robomimic || git clone -q https://github.com/ARISE-Initiative/robomimic.git $(UPSTREAM)/robomimic
	@test -d $(UPSTREAM)/robosuite || git clone -q https://github.com/ARISE-Initiative/robosuite.git $(UPSTREAM)/robosuite
	git -C $(UPSTREAM)/robomimic checkout -q $(ROBOMIMIC_COMMIT)
	git -C $(UPSTREAM)/robosuite checkout -q $(ROBOSUITE_TAG)
	@test -d .venv || uv venv -q --python 3.11 .venv
	uv pip install -q --python $(PYTHON) cmake
	@# Install exactly the tested set without re-resolving it: the lock pins mujoco 3.2.7, older than
	@# an unused robosuite extra (mink) declares. egl_probe's build file predates CMake 4.
	$(VENV_PATH) CMAKE_POLICY_VERSION_MINIMUM=3.5 uv pip install -q --python $(PYTHON) --no-deps -r requirements-lock.txt
	uv pip install -q --python $(PYTHON) --no-deps -e $(UPSTREAM)/robosuite -e $(UPSTREAM)/robomimic
	@$(PYTHON) -c "import robomimic, robosuite, mujoco; print('ready: robomimic', robomimic.__version__, '| robosuite', robosuite.__version__, '| mujoco', mujoco.__version__)"

test:  ## run the unit tests (no dataset or simulator needed)
	$(PYTHON) -m pytest tests -q

lint:  ## check style and formatting
	$(PYTHON) -m ruff check scripts tests
	$(PYTHON) -m ruff format --check scripts tests

check:  ## recount every recorded result from the per-trial records and redraw the charts
	$(PYTHON) -m scripts.report

reproduce-quick:  ## rerun the first model end to end and compare (about 5 minutes)
	PYTHON=$(PYTHON) scripts/reproduce.sh --quick reproduction-quick

reproduce:  ## rerun everything except the long exploration runs and compare (about 47 minutes)
	PYTHON=$(PYTHON) scripts/reproduce.sh

reproduce-full:  ## rerun everything, including the long exploration runs
	PYTHON=$(PYTHON) scripts/reproduce.sh --full
