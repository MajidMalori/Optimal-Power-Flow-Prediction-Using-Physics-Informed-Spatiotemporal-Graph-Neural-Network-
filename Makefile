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

# Runs only the topology reconfiguration tests
test-topology:
	python3 tests/test_topology.py

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

# Generate 96 timestep data for case 33:
gen-33:
	python scripts/generate_data.py --case 33 --timestep 96

gen-57:
	python scripts/generate_data.py --case 57 --timestep 96

gen-118:
	python scripts/generate_data.py --case 118 --timestep 96

gen-all:
	python scripts/generate_data.py --case all --timestep 96

gen-full:
	python scripts/generate_data.py --timestep 10008

prep-33:
	python scripts/preprocess_data.py --case 33

prep-57:
	python scripts/preprocess_data.py --case 57

prep-118:
	python scripts/preprocess_data.py --case 118

prep-all:
	python scripts/preprocess_data.py --case all

# Train model for case 33:
train-33:
	python scripts/train.py --case 33 --models all

train-57:
	python scripts/train.py --case 57 --models all

train-118:
	python scripts/train.py --case 118 --models all

train-all:
	python scripts/train.py --case all --models all

# Train model for case 33 online:
train-33-online:
	python scripts/train.py --case 33 --models all --online

# Sync wandb logs to cloud
sync:
	wandb sync wandb_logs/wandb/offline-run-*

# Generate animations for all fractions
anim-33:
	python scripts/animate_grid_dynamics.py --case case33

anim-57:
	python scripts/animate_grid_dynamics.py --case case57

anim-118:
	python scripts/animate_grid_dynamics.py --case case118

anim-all:
	python scripts/animate_grid_dynamics.py --case all

# Evaluate trained models on the test set
eval-33:
	python scripts/evaluate.py --case case33

eval-57:
	python scripts/evaluate.py --case case57

eval-118:
	python scripts/evaluate.py --case case118

eval-all:
	python scripts/evaluate.py --case all

# Analyze uncertainty on the test set
unc-33:
	python scripts/analyze_uncertainty.py --case case33

unc-57:
	python scripts/analyze_uncertainty.py --case case57

unc-118:
	python scripts/analyze_uncertainty.py --case case118

unc-all:
	python scripts/analyze_uncertainty.py --case all


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
	rm -rf data/01_raw/*
	rm -rf data/03_processed/*
	rm -rf reports/figures/01_raw_data/*
	rm -rf reports/figures/03_processed/*
	rm -rf reports/animations/*
