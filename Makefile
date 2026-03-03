# Python project Makefile
.PHONY: test test-fast test-physics test-models clean

# The default Python interpreter to use
PYTHON = python

# Default test command - runs everything in verbose mode
test:
	pytest

# Runs tests but stops on the very first failure (fail fast)
test-fast:
	pytest -x

# Runs only the physical data validation tests
test-physics:
	pytest tests/test_data_physics.py

# Runs only the neural network model tests
test-models:
	pytest tests/test_models.py

# Runs only the data preprocessing tests
test-preprocessing:
	pytest tests/test_preprocessing.py

# Clean up temporary Python files and cache
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
