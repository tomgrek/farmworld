VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
POETRY = $(VENV)/bin/poetry

venv:
	python -m venv $(VENV)

install:
	. $(VENV)/bin/activate
	$(PIP) install poetry
	$(POETRY) install --no-root

tidy:
	. $(VENV)/bin/activate
	$(PYTHON) -m black .

lint:
	. $(VENV)/bin/activate
	$(PYTHON) -m black . --check