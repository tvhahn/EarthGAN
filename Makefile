.PHONY: clean data requirements

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = earthgan
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Create environment and install dependencies
create_environment: test_environment
ifeq (True,$(HAS_CONDA)) # assume on local
	@echo ">>> Detected conda. Assume local computer. Installing packages from yml."
	conda env create -f envearth.yml
else # assume on HPC
	@echo ">>> No Conda detected. Assume on HPC. Creating venv with Jupyter Lab functionality."
	bash bash_scripts/make_hpc_venv.sh $(PROJECT_DIR)
	@echo ">>> venv created. Activate with source ~/earth/bin/activate"
endif

## Download data
download:
ifeq (True,$(HAS_CONDA)) # assume on local
	bash src/data/download_data_local.sh $(PROJECT_DIR)
else # assume on HPC
	bash src/data/download_data_hpc.sh $(PROJECT_DIR)
endif
	

## Extract downloaded data and rename directories as needed
extract:
ifeq (True,$(HAS_CONDA)) # assume on local
	bash src/data/extract_data_local.sh $(PROJECT_DIR)
else # assume on HPC
	bash src/data/extract_data_hpc.sh $(PROJECT_DIR)
endif


## Make Dataset
data: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	bash bash_scripts/make_local_data.sh $(PROJECT_DIR)
else # assume on HPC
	sbatch bash_scripts/make_hpc_data.sh $(PROJECT_DIR)
endif


## Train the models
train:
ifeq (True,$(HAS_CONDA)) # assume on local
	bash bash_scripts/train_model_local.sh $(PROJECT_DIR)
else # assume on HPC
	sbatch bash_scripts/train_model_hpc.sh $(PROJECT_DIR)
endif


## Make the figures of the results
figures_results:
	$(PYTHON_INTERPRETER) src/visualization/visualize_results.py


## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
