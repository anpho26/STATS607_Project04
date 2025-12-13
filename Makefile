PYTHON := .venv/bin/python

.PHONY: help venv install \
        figs figs-demo \
        figs-convergence \
        figs-model-selection figs-model-selection-parallel \
        profile clean

# -------------------------------------------------------------------
# Help
# -------------------------------------------------------------------
help:
	@echo "Available targets:"
	@echo "  venv                       Create .venv virtual environment"
	@echo "  install                    Install dependencies into .venv"
	@echo "  figs                       Run all figure-generating scripts"
	@echo "  figs-demo                  Run synthetic demo (dendrogram + DIC)"
	@echo "  figs-convergence           Run convergence experiments (strong + weak)"
	@echo "  figs-model-selection       Run serial model-selection experiments"
	@echo "  figs-model-selection-parallel  Run parallel model-selection experiments"
	@echo "  profile                    Compare serial vs parallel runtimes"
	@echo "  clean                      Remove generated outputs and caches"

# -------------------------------------------------------------------
# Environment / dependencies
# -------------------------------------------------------------------
venv:
	python -m venv .venv

install: venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# -------------------------------------------------------------------
# Figure generation
# -------------------------------------------------------------------

# Synthetic demo: over-fitted GMM, dendrogram, DIC along dendrogram
figs-demo: install
	$(PYTHON) examples/demo_synthetic.py

# Convergence experiments (strongly and weakly identifiable regimes)
figs-convergence: install
	$(PYTHON) examples/experiment_convergence_strong.py
	$(PYTHON) examples/experiment_convergence_weak.py

# Serial model-selection experiments:
#   - Well-specified Gaussian mixture
#   - ε-contamination
#   - Skew-normal mixture
# Each script also produces panels (a),(b),(c),(d) for all three regimes.
figs-model-selection: install
	$(PYTHON) examples/experiment_model_selection.py

# Parallel model-selection experiments (same regimes, joblib-powered)
figs-model-selection-parallel: install
	$(PYTHON) examples/experiment_model_selection_parallel.py

# Run ALL figure-generating scripts
figs: figs-demo figs-convergence figs-model-selection figs-model-selection-parallel

# -------------------------------------------------------------------
# Profiling
# -------------------------------------------------------------------
profile: install
	$(PYTHON) examples/profile_model_selection.py

# -------------------------------------------------------------------
# Clean generated outputs
# -------------------------------------------------------------------
clean:
	# Figures + experiment outputs
	rm -rf out/ figures/
	# Python caches
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.py[co]" -delete
	# Test / coverage caches (if any)
	rm -rf .pytest_cache/ .coverage htmlcov/