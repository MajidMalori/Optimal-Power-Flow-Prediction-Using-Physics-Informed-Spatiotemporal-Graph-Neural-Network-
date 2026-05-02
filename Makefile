# Python project Makefile
.PHONY: test test-fast test-physics test-models clean pylint

# Interpreter selection (override from CLI or env if needed):
#   make test PYTHON_MAIN=.venv/Scripts/python
PYTHON_MAIN ?= python
PYTEST ?= $(PYTHON_MAIN) -m pytest

# Default test command - runs everything in verbose mode
test:
	$(PYTEST)

# Runs tests but stops on the very first failure (fail fast)
test-fast:
	$(PYTEST) -x

# Runs only the physical data validation tests
test-physics:
	$(PYTEST) tests/test_data_physics.py

# Runs only the neural network model tests
test-models:
	$(PYTEST) tests/test_models.py

# Runs only the topology reconfiguration tests
test-topology:
	$(PYTHON_MAIN) tests/test_topology.py

# Runs only the end-to-end training tests
test-e2e:
	$(PYTEST) tests/test_training_e2e.py

# Runs only the end-to-end warmstart tests
test-warmstart:
	$(PYTEST) tests/test_warmstart_e2e.py -v -s

# Runs only the data preprocessing tests
test-preprocessing:
	$(PYTEST) tests/test_preprocessing.py

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
	$(PYTHON_MAIN) scripts/generate_data.py --case 33 --timestep 96

gen-57:
	$(PYTHON_MAIN) scripts/generate_data.py --case 57 --timestep 96

gen-118:
	$(PYTHON_MAIN) scripts/generate_data.py --case 118 --timestep 96

gen-all:
	$(PYTHON_MAIN) scripts/generate_data.py --case all --timestep 96

gen-full:
	$(PYTHON_MAIN) scripts/generate_data.py --timestep 10008

prep-33:
	$(PYTHON_MAIN) scripts/preprocess_data.py --case 33

prep-57:
	$(PYTHON_MAIN) scripts/preprocess_data.py --case 57

prep-118:
	$(PYTHON_MAIN) scripts/preprocess_data.py --case 118

prep-all:
	$(PYTHON_MAIN) scripts/preprocess_data.py --case all

# Train model for case 33:
train-33:
	$(PYTHON_MAIN) scripts/train.py --case 33 --models all

train-57:
	$(PYTHON_MAIN) scripts/train.py --case 57 --models all

train-118:
	$(PYTHON_MAIN) scripts/train.py --case 118 --models all

train-all:
	$(PYTHON_MAIN) scripts/train.py --case all --models all

# Train model for case 33 online:
train-33-online:
	$(PYTHON_MAIN) scripts/train.py --case 33 --models all --online

# Sync wandb logs to cloud
sync:
	wandb sync wandb_logs/wandb/offline-run-*

# Generate animations for all fractions
anim-33:
	$(PYTHON_MAIN) scripts/animate_grid_dynamics.py --case case33

anim-57:
	$(PYTHON_MAIN) scripts/animate_grid_dynamics.py --case case57

anim-118:
	$(PYTHON_MAIN) scripts/animate_grid_dynamics.py --case case118

anim-all:
	$(PYTHON_MAIN) scripts/animate_grid_dynamics.py --case all

# Evaluate trained models on the test set
eval-33:
	$(PYTHON_MAIN) scripts/evaluate.py --case case33

eval-57:
	$(PYTHON_MAIN) scripts/evaluate.py --case case57

eval-118:
	$(PYTHON_MAIN) scripts/evaluate.py --case case118

eval-all:
	$(PYTHON_MAIN) scripts/evaluate.py --case all

# Analyze uncertainty on the test set
unc-33:
	$(PYTHON_MAIN) scripts/analyze_uncertainty.py --case case33

unc-57:
	$(PYTHON_MAIN) scripts/analyze_uncertainty.py --case case57

unc-118:
	$(PYTHON_MAIN) scripts/analyze_uncertainty.py --case case118

unc-all:
	$(PYTHON_MAIN) scripts/analyze_uncertainty.py --case all

# Warm-Start benchmarking runs
warmstart-33:
	$(PYTHON_MAIN) scripts/benchmark_warmstart.py --case case33 --model all

warmstart-57:
	$(PYTHON_MAIN) scripts/benchmark_warmstart.py --case case57 --model all

warmstart-118:
	$(PYTHON_MAIN) scripts/benchmark_warmstart.py --case case118 --model all

warmstart-all:
	$(PYTHON_MAIN) scripts/benchmark_warmstart.py --case all --model all

# MoSOA Benchmarks:
math:
	$(PYTHON_MAIN) scripts/benchmark_math.py

pert:
	$(PYTHON_MAIN) scripts/benchmark_perturbation.py


tune-33:
	$(PYTHON_MAIN) scripts/benchmark_hpo_tuning.py --case case33 --all-models

tune-57:
	$(PYTHON_MAIN) scripts/benchmark_hpo_tuning.py --case case57 --all-models

tune-118:
	$(PYTHON_MAIN) scripts/benchmark_hpo_tuning.py --case case118 --all-models

# Expanded Visualization and Mathematical HPO
landscapes:
	$(PYTHON_MAIN) scripts/visualize_landscapes.py

tune-math:
	$(PYTHON_MAIN) scripts/benchmark_math_hpo.py

sense:
	$(PYTHON_MAIN) scripts/analyze_sensitivity.py

# Warm-start benchmark pipeline (new modular scripts)
gen-bench-33:
	$(PYTHON_MAIN) scripts/generate_benchmark_states.py --case case33

gen-bench-57:
	$(PYTHON_MAIN) scripts/generate_benchmark_states.py --case case57

gen-bench-118:
	$(PYTHON_MAIN) scripts/generate_benchmark_states.py --case case118

gen-bench-all:
	$(PYTHON_MAIN) scripts/generate_benchmark_states.py --case all

ws-speed-33:
	$(PYTHON_MAIN) scripts/benchmark_ws_speed.py --case case33

ws-speed-57:
	$(PYTHON_MAIN) scripts/benchmark_ws_speed.py --case case57

ws-speed-118:
	$(PYTHON_MAIN) scripts/benchmark_ws_speed.py --case case118

ws-feas-33:
	$(PYTHON_MAIN) scripts/benchmark_ws_feasibility.py --case case33

ws-feas-57:
	$(PYTHON_MAIN) scripts/benchmark_ws_feasibility.py --case case57

ws-feas-118:
	$(PYTHON_MAIN) scripts/benchmark_ws_feasibility.py --case case118

ws-rescue-33:
	$(PYTHON_MAIN) scripts/benchmark_ws_rescue.py --case case33

ws-rescue-57:
	$(PYTHON_MAIN) scripts/benchmark_ws_rescue.py --case case57

ws-rescue-118:
	$(PYTHON_MAIN) scripts/benchmark_ws_rescue.py --case case118

ws-core-33: gen-bench-33 ws-speed-33 ws-feas-33
ws-core-57: gen-bench-57 ws-speed-57 ws-feas-57
ws-core-118: gen-bench-118 ws-speed-118 ws-feas-118

ws-all-33: gen-bench-33 ws-speed-33 ws-feas-33 ws-rescue-33
ws-all-57: gen-bench-57 ws-speed-57 ws-feas-57 ws-rescue-57
ws-all-118: gen-bench-118 ws-speed-118 ws-feas-118 ws-rescue-118

ws-smoke-33:
	$(PYTHON_MAIN) scripts/generate_benchmark_states.py --case case33 --max-samples 5
	$(PYTHON_MAIN) scripts/benchmark_ws_speed.py --case case33 --max-samples 5
	$(PYTHON_MAIN) scripts/benchmark_ws_feasibility.py --case case33 --max-samples 5
	$(PYTHON_MAIN) scripts/benchmark_ws_rescue.py --case case33 --max-samples 5

# Full pipeline for all cases
full-33: gen-33 prep-33 test train-33 eval-33 unc-33
full-57: gen-57 prep-57 test train-57 eval-57 unc-57
full-118: gen-118 prep-118 test train-118 eval-118 unc-118

# End-to-end pipeline (Pipeline Orchestration)
full-test: clean-training clean gen-all prep-all test train-all eval-all unc-all


# Clean up temporary Python files and cache
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	find . -type d -name "wandb" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Clean up training logs (local)
clean-logs:
	@rm -rf logs/

# Clean up model checkpoints
clean-checkpoints:
	@rm -rf checkpoints/

# Clean up W&B logs
clean-wandb:
	@rm -rf wandb_logs/

# Clean up ALL training sessions (checkpoints and logs)
clean-training: clean-logs clean-checkpoints clean-wandb

# Clean up only report directories
clean-reports:
	@rm -rf reports/raw_data/
	@rm -rf reports/prep_data/
	@rm -rf reports/animations/
	@rm -rf reports/evaluation/
	@rm -rf reports/figures/
	@rm -rf reports/training/

# Clean up raw and processed data
clean-data-raw:
	@rm -rf data/raw/*

clean-data-processed:
	@rm -rf data/prep/*

# Comprehensive clean-up (code, data, and training)
clean-all: clean clean-training clean-reports clean-data-raw clean-data-processed
