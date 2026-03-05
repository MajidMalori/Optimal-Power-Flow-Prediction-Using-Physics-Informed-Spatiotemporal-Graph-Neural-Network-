# Python project Makefile
.PHONY: test test-fast test-physics test-models clean pylint

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

# Runs only the end-to-end training tests
test-e2e:
	pytest tests/test_training_e2e.py

# Runs only the data preprocessing tests
test-preprocessing:
	pytest tests/test_preprocessing.py

# Runs pylint on all Python files (full check)
pylint:
	pylint src/data/*.py src/models/*.py tests/*.py

# Categorized Linting
lint-imports:
	pylint --disable=all --enable=imports,wrong-import-order,unused-import,reimported,cyclic-import data/*.py models/*.py tests/*.py

lint-unused:
	pylint --disable=all --enable=unused-variable,unused-argument,unused-import,unused-wildcard-import data/*.py models/*.py tests/*.py

lint-duplication:
	pylint --disable=all --enable=duplicate-code data/*.py models/*.py tests/*.py

lint-naming:
	pylint --disable=all --enable=invalid-name,blacklisted-name data/*.py models/*.py tests/*.py

lint-complexity:
	pylint --disable=all --enable=too-many-branches,too-many-statements,too-many-locals,too-many-arguments data/*.py models/*.py tests/*.py

# Generate 120 timestep data for all cases
main-120:
	python src/data/main.py --case all --timestep 120

# Generate 96 timestep data for case 33:
main-test:
	python src/data/main.py --case 33 --timestep 96

# Preprocess data for case 33:
pre-test:
	python src/data/preprocess_data.py --case 33

# Train model for case 33:
train-test:
	python scripts/train.py --case 33 --models all

# Train model for case 33 online:
train-test-online:
	python scripts/train.py --case 33 --models all --online

# Sync wandb logs to cloud
sync:
	wandb sync wandb_logs/wandb/offline-run-*

# Clean up temporary Python files and cache
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	find . -type d -name "wandb" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Clean up ALL training sessions (checkpoints and logs)
clean-training:
	rm -rf checkpoints/
	rm -rf wandb_logs/

# Comprehensive clean-up (code, data, and training)
clean-all: clean clean-training
	rm -rf src/data/01_raw/
	rm -rf src/data/03_processed/
	rm -rf reports/figures/01_raw_data/
	rm -rf reports/figures/03_processed/
