.PHONY: run test lint fmt clean install dev

# Run the Streamlit app
run:
	streamlit run app/main.py

# Run tests
test:
	pytest tests/ -v

# Run tests with coverage
coverage:
	pytest tests/ --cov=core --cov=data --cov=history --cov=app --cov-report=term-missing

# Lint code
lint:
	ruff check .

# Format code
fmt:
	ruff format .
	ruff check --fix .

# Install dependencies
install:
	pip install -e .

# Install with dev dependencies
dev:
	pip install -e ".[dev]"

# Install with Bloomberg provider
install-bloomberg:
	pip install -e ".[bloomberg]"

# Install with Interactive Brokers provider
install-ib:
	pip install -e ".[ib]"

# Install everything (dev + all providers)
install-all:
	pip install -e ".[dev,bloomberg,ib]"

# Clean caches and build artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist build .pytest_cache htmlcov .coverage .ruff_cache
