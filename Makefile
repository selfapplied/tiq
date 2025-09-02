# TIQ - Timeline Interleaver & Quantizer Makefile

.PHONY: help install install-dev test lint format clean build docs

# Default target
help:
	@echo "TIQ - Timeline Interleaver & Quantizer"
	@echo ""
	@echo "Available targets:"
	@echo "  install     Install TIQ and dependencies"
	@echo "  install-dev Install TIQ with development dependencies"
	@echo "  test        Run test suite"
	@echo "  lint        Run linting checks"
	@echo "  format      Format code with black"
	@echo "  clean       Clean build artifacts"
	@echo "  build       Build package"
	@echo "  docs        Generate documentation"
	@echo "  demo        Run TIQ demo"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,crypto]"

# Testing
test:
	python -m pytest test_tiq.py -v

test-coverage:
	python -m pytest test_tiq.py --cov=tiq --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 tiq.py test_tiq.py
	mypy tiq.py

format:
	black tiq.py test_tiq.py

# Build
build:
	python -m build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Documentation
docs:
	@echo "Documentation is in README.md"
	@echo "CE1 specification is in tiq.ce1"

# Demo
demo:
	@echo "⧗ TIQ Demo"
	@echo ""
	@echo "Creating demo repositories..."
	@mkdir -p demo/repo1 demo/repo2 demo/plain
	@cd demo/repo1 && git init && echo "content1" > file1.txt && git add . && git commit -m "initial commit"
	@cd demo/repo2 && git init && echo "content2" > file2.txt && git add . && git commit -m "initial commit"
	@echo "plain content" > demo/plain/plain_file.txt
	@echo ""
	@echo "Running TIQ superpose..."
	@cd demo && python ../tiq.py superpose --apply --include-non-git
	@echo ""
	@echo "Mapping branches..."
	@cd demo && python ../tiq.py map
	@echo ""
	@echo "Demo complete! Check demo/ directory for results."
	@echo "Clean up with: make clean-demo"

clean-demo:
	rm -rf demo/

# Development helpers
dev-setup: install-dev
	@echo "Development environment ready!"
	@echo "Run 'make test' to verify installation"

check: lint test
	@echo "All checks passed!"

# CI/CD helpers
ci-test:
	python -m pytest test_tiq.py --junitxml=test-results.xml

ci-lint:
	flake8 tiq.py test_tiq.py --output-file=flake8-results.txt

# Version management
version:
	@python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"

# Help for specific targets
test-help:
	@echo "Test targets:"
	@echo "  test           Run all tests"
	@echo "  test-coverage  Run tests with coverage report"
	@echo "  ci-test        Run tests for CI (with XML output)"

lint-help:
	@echo "Linting targets:"
	@echo "  lint           Run flake8 and mypy"
	@echo "  format         Format code with black"
	@echo "  ci-lint        Run linting for CI (with file output)"
