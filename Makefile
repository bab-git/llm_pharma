# LLM Pharma Makefile
.PHONY: help install dev-install run run-backend run-frontend lint format test test-unit test-integration check pre-commit clean setup-data setup-patients setup-policies setup-trials dev-setup quick-check

# =====================
# Help & Documentation
# =====================
help:
	@echo "Available commands:"
	@echo "  install            - Install main dependencies and setup Poetry venv (.lpenv)"
	@echo "  dev-install        - Install all dependencies including dev tools"
	@echo ""
	@echo "Run Commands:"
	@echo "  run                - Run the Gradio frontend dashboard"
	@echo ""
	@echo "Testing:"
	@echo "  test-all           - Run all tests"
	@echo "  test-unit          - Run unit tests only"
	@echo "  test-integration   - Run integration tests only"
	@echo "  test-regression    - Run a specific standalone end-to-end regression test"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint               - Run linting (flake8, mypy)"
	@echo "  format             - Format code (black, isort)"
	@echo "  verify             - Run all checks (lint + format + test-all)"
	@echo ""
	@echo "Other:"
	@echo "  pre-commit         - Setup pre-commit hooks"
	@echo "  clean              - Clean up build artifacts and cache"
	@echo "  setup-data         - Set up all data (patients, policies, trials)"



# =====================
# Poetry Environment
# =====================
install:
	@echo "Installing dependencies with Poetry in .lpenv..."
	poetry config virtualenvs.in-project true --local
	poetry env use $(shell cat .python-version | tr -d '\n')
	poetry install --only main
	@echo "Installation complete!"

dev-install:
	@echo "Installing all dependencies including dev tools..."
	poetry install
	@echo "Development installation complete!"

# =====================
# Run Commands
# =====================
run:
	@echo "Starting Gradio frontend dashboard..."
	poetry run python frontend/app.py

# =====================
# Data Setup Scripts
# =====================
setup-data:
	@echo "Setting up all data for LLM Pharma system..."
	poetry run python scripts/setup_all_data.py --force-recreate

# =====================
# Code Quality
# =====================
PY_SRC := backend frontend tests scripts

lint:
	@echo "Running linting checks..."
	poetry run ruff check $(PY_SRC) --fix

format:
	@echo "Formatting code..."
	poetry run black $(PY_SRC)
	poetry run isort $(PY_SRC)
	@echo "Code formatting complete!"

# =====================
# Testing
# =====================
test-all: test-unit test-integration test-regression
	@echo "All tests completed!"

test-unit:
	@echo "Running unit tests only..."
	poetry run pytest tests/unit/ -v

test-integration:
	@echo "Running integration tests only..."
	poetry run pytest tests/integration/ -v

test-regression:
	@echo "Running standalone regression tests..."
	poetry run python tests/regression/test_end_to_end.py

# =====================
# Combined Checks
# =====================
verify: lint format test-all
	@echo "All checks completed!"

# =====================
# Cleanup
# =====================
clean:
	@echo "Cleaning up build artifacts and cache..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	@echo "Cleanup complete!"

# =====================
# Development Workflow
# =====================
dev-setup: dev-install pre-commit
	@echo "Development environment setup complete!"

quick-check:
	@echo "Running quick checks (format + lint)..."
	poetry run black --check backend/ frontend/ tests/
	poetry run isort --check-only backend/ frontend/ tests/
	poetry run flake8 backend/ frontend/ tests/
	@echo "Quick checks complete!"

pre-commit:
	@echo "Setting up pre-commit hooks..."
	poetry run pre-commit install
	@echo "Pre-commit hooks installed!"
