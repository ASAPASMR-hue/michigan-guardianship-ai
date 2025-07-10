.PHONY: help install test lint format clean run-orchestrator validate-configs setup-dev docs serve-docs

# Default target
help:
	@echo "Michigan Minor Guardianship AI - Available Commands"
	@echo "=================================================="
	@echo "Development:"
	@echo "  make install        - Install all dependencies"
	@echo "  make setup-dev      - Set up development environment"
	@echo "  make test           - Run all tests"
	@echo "  make lint           - Run linting checks"
	@echo "  make format         - Auto-format code"
	@echo "  make type-check     - Run type checking"
	@echo ""
	@echo "Operations:"
	@echo "  make run-orchestrator - Run the Gemini orchestrator"
	@echo "  make validate-configs - Validate all YAML configurations"
	@echo "  make update-embeddings - Update document embeddings"
	@echo "  make benchmark       - Run performance benchmarks"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs           - Build documentation"
	@echo "  make serve-docs     - Serve documentation locally"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          - Remove temporary files"
	@echo "  make security-check - Run security scan"
	@echo "  make update-deps    - Update dependencies"

# Python interpreter
PYTHON := python3
PIP := $(PYTHON) -m pip

# Virtual environment
VENV := venv
VENV_ACTIVATE := . $(VENV)/bin/activate

# Install dependencies
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Set up development environment
setup-dev:
	$(PYTHON) -m venv $(VENV)
	$(VENV_ACTIVATE) && $(PIP) install --upgrade pip
	$(VENV_ACTIVATE) && $(PIP) install -r requirements.txt
	$(VENV_ACTIVATE) && $(PIP) install -e .
	@echo "Development environment ready. Activate with: source $(VENV)/bin/activate"

# Run tests
test:
	pytest tests/ -v --cov=scripts --cov-report=html --cov-report=term

# Linting
lint:
	flake8 scripts/ --max-line-length=120 --extend-ignore=E203,W503
	mypy scripts/ --ignore-missing-imports
	black --check scripts/
	isort --check-only scripts/

# Auto-format code
format:
	black scripts/
	isort scripts/

# Type checking
type-check:
	mypy scripts/ --ignore-missing-imports

# Run the orchestrator
run-orchestrator:
	@if [ -z "$$GEMINI_API_KEY" ]; then \
		echo "Error: GEMINI_API_KEY environment variable not set"; \
		exit 1; \
	fi
	$(PYTHON) scripts/orchestrator.py 2>&1 | tee logs/run_$$(date +%F_%H%M%S).log

# Validate all configurations
validate-configs:
	@echo "Validating YAML configurations..."
	@$(PYTHON) -c "import yaml, sys; \
		from pathlib import Path; \
		failed = False; \
		for f in Path('.').rglob('*.yaml'): \
			if '.github' not in str(f): \
				try: \
					yaml.safe_load(open(f)); \
					print(f'✓ {f}'); \
				except Exception as e: \
					print(f'✗ {f}: {e}'); \
					failed = True; \
		sys.exit(1 if failed else 0)"
	@echo "Validating JSON configurations..."
	@$(PYTHON) -c "import json, sys; \
		from pathlib import Path; \
		failed = False; \
		for f in Path('.').rglob('*.json'): \
			try: \
				json.load(open(f)); \
				print(f'✓ {f}'); \
			except Exception as e: \
				print(f'✗ {f}: {e}'); \
				failed = True; \
		sys.exit(1 if failed else 0)"

# Update embeddings
update-embeddings:
	@echo "Updating document embeddings..."
	$(PYTHON) scripts/update_embeddings.py

# Run benchmarks
benchmark:
	@echo "Running performance benchmarks..."
	$(PYTHON) scripts/run_benchmarks.py

# Build documentation
docs:
	mkdocs build

# Serve documentation
serve-docs:
	mkdocs serve --dev-addr localhost:8001

# Clean temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name ".DS_Store" -delete
	rm -rf .coverage htmlcov .pytest_cache .mypy_cache
	rm -rf build dist *.egg-info

# Security check
security-check:
	@echo "Running security checks..."
	# Check for hardcoded secrets
	@if grep -r "AIzaSy" --include="*.py" --include="*.yaml" . 2>/dev/null; then \
		echo "WARNING: Potential API key found in code"; \
		exit 1; \
	fi
	# Run safety check on dependencies
	safety check || true

# Update dependencies
update-deps:
	$(PIP) list --outdated
	@echo ""
	@echo "To update a specific package: pip install --upgrade package_name"
	@echo "To update all: pip install --upgrade -r requirements.txt"

# CI/CD simulation
ci-local:
	@echo "Running local CI checks..."
	make lint
	make test
	make validate-configs
	make security-check
	@echo "✓ All CI checks passed"

# Initialize git repository
git-init:
	@if [ ! -d .git ]; then \
		git init; \
		git add .; \
		git commit -m "Initial commit: Michigan Minor Guardianship AI project structure"; \
		echo "✓ Git repository initialized"; \
	else \
		echo "Git repository already exists"; \
	fi

# Check project structure
check-structure:
	@echo "Checking project structure..."
	@for dir in config constants patterns samples src scripts tests rubrics data logs results; do \
		if [ -d "$$dir" ]; then \
			echo "✓ $$dir/"; \
		else \
			echo "✗ $$dir/ (missing)"; \
		fi \
	done
	@for file in requirements.txt Makefile .gitignore; do \
		if [ -f "$$file" ]; then \
			echo "✓ $$file"; \
		else \
			echo "✗ $$file (missing)"; \
		fi \
	done