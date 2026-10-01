# money-printer-research

[![Release](https://img.shields.io/github/v/release/Ax-47/money-printer-research)](https://img.shields.io/github/v/release/Ax-47/money-printer-research)
[![Build status](https://img.shields.io/github/actions/workflow/status/Ax-47/money-printer-research/main.yml?branch=main)](https://github.com/Ax-47/money-printer-research/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/Ax-47/money-printer-research/branch/main/graph/badge.svg)](https://codecov.io/gh/Ax-47/money-printer-research)
[![Commit activity](https://img.shields.io/github/commit-activity/m/Ax-47/money-printer-research)](https://img.shields.io/github/commit-activity/m/Ax-47/money-printer-research)
[![License](https://img.shields.io/github/license/Ax-47/money-printer-research)](https://img.shields.io/github/license/Ax-47/money-printer-research)

A short description of the project.

- **Github repository**: <https://github.com/Ax-47/money-printer-research/>
- **Documentation** <https://Ax-47.github.io/money-printer-research/>

## Getting started with your project

### 1. Create a New Repository

First, create a repository on GitHub with the same name as this project, and then run the following commands locally:

```bash
git init -b main
git add .
git commit -m "init commit"
git remote add origin git@github.com:Ax-47/money-printer-research.git
git push -u origin main
```

### 2. Set Up Your Development Environment

Then, install the environment and optional run tools depending on your cookiecutter choices.

```bash
make install
```

This will also generate your `uv.lock` file

### 3. Run the pre-commit hooks

Initially, the CI pipeline might be failing due to formatting issues. To resolve those run:

```bash
uv run pre-commit run -a
```

### 4. Commit the changes

Lastly, commit the changes made by the two steps above to your repository.

```bash
git add .
git commit -m 'Fix formatting issues'
git push origin main
```

You are now ready to start development on your project!
The CI pipeline will be triggered when you open a pull request, merge to main, or when you create a new release.

To finalize the set-up for publishing to PyPI, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/codecov/).

## Releasing a new version

<!--  -->

- Create an API Token on [PyPI](https://pypi.org/).
- Add the API Token to your projects secrets with the name `PYPI_TOKEN` by visiting [this page](https://github.com/Ax-47/money-printer-research/settings/secrets/actions/new).
- Create a [new release](https://github.com/Ax-47/money-printer-research/releases/new) on Github.
- Create a new tag in the form `*.*.*`.

For more details, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/cicd/#how-to-trigger-a-release).
<!--  -->

---

Repository initiated with [tiefenthaler/uv-datascience-project-template](https://github.com/tiefenthaler/uv-datascience-project-template).
# money-printer-research
