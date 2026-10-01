.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using uv"
	@uv sync --all-groups

.PHONY: install-hooks
install-hooks: ## Install the pre-commit hooks
	@echo "🚀 Installing pre-commit hooks"
	@uv run pre-commit install

.PHONY: format
format: ## Format code
	@echo "🚀 Formatting code: Running ruff"
	@uv run ruff format

.PHONY: lint-fix
lint-fix: ## Lint and fix code
	@echo "🚀 Linting and fixing code: Running ruff"
	@uv run ruff check --fix

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting code: Running ruff"
	@uv run ruff check
	@echo "🚀 Static type checking: Running pyright"
	@uv run pyright
	@echo "🚀 Installing pre-commit hooks"
	@uv run pre-commit install
	@echo "🚀 Running pre-commit"
	@uv run pre-commit run -a

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@uv run pytest --cov

.PHONY: run-api
run-api: ## Run the FastAPI application
	@echo "🚀 Running the FastAPI application"
	@uv run python -m money_printer_research.app_fastapi_autoencoder

.PHONY: build
build: clean-build ## Build wheel file
	@echo "🚀 Creating wheel file"
	@uvx --from build pyproject-build --installer uv

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "🚀 Removing build artifacts"
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

# 
.PHONY: publish
publish: ## Publish a release to PyPI.
	@echo "🚀 Publishing."
	@uvx twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

.PHONY: build-and-publish
build-and-publish: build publish ## Build and publish.
# 
.PHONY: docs-build
docs-build: ## Test if documentation can be built without warnings or errors
	@uv run mkdocs build -s

.PHONY: docs-serve
docs-serve: ## Build and serve the documentation
	@uv run mkdocs serve

.PHONY: clean
clean: ## Clean all artifacts and remove virtual environment
	@echo "🚀 Cleaning all artifacts and removing virtual environment"
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None; \
	shutil.rmtree('build') if os.path.exists('build') else None; \
	shutil.rmtree('docs/site') if os.path.exists('docs/site') else None; \
	shutil.rmtree('.pytest_cache') if os.path.exists('.pytest_cache') else None; \
	shutil.rmtree('.ruff_cache') if os.path.exists('.ruff_cache') else None"
	@uv run python -c "import shutil; shutil.rmtree('.venv', ignore_errors=True)"

.PHONY: update-packages
update-packages: ## Update all Python packages to the latest versions of the `uv.lock` file and sync the environment
	@echo "🚀 Updating all Python packages..."
	@uv lock --upgrade
	@uv sync --all-groups
	@echo "✅ Packages updated and environment is in sync."

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^[a-zA-Z_-]+:.*?## (.*)$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
