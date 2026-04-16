# Contributing to CIPA

Thank you for your interest in contributing. This document covers how to set
up a development environment, run tests, and submit changes.

---

## Development setup

```bash
git clone <repository-url>
cd cipa
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.11, pip ≥ 23.

---

## Running tests

```bash
pytest                      # all tests, ≥ 90% coverage required
pytest tests/unit/          # unit tests only
pytest -k "test_d1"         # filter by name
pytest --no-cov             # skip coverage (faster for exploration)
```

The test suite must pass before any pull request is merged. The 90% coverage
gate is enforced by the CI pipeline.

---

## Code style

CIPA uses [ruff](https://docs.astral.sh/ruff/) for both linting and
formatting. All checks must pass before submission.

```bash
ruff check src/             # lint
ruff format src/            # auto-format
ruff check --fix src/       # lint + auto-fix safe issues
```

Key conventions already configured in `pyproject.toml`:
- Line length: 88 characters.
- Quotes: double.
- Mathematical variable names (`X`, `N`, `IR`, `D1`–`D7`) are allowed without
  triggering naming warnings.

---

## Submitting changes

1. Open an issue describing the bug or feature before starting significant
   work, to avoid duplicated effort.
2. Fork the repository and create a branch from `main`.
3. Make your changes with tests covering the new behaviour.
4. Ensure `ruff check src/` and `pytest` both pass locally.
5. Submit a pull request referencing the relevant issue.

---

## Reporting issues

Please use the GitHub issue tracker. Include:
- Python version (`python --version`).
- CIPA version (`python -c "import cipa; print(cipa.__version__)"`).
- A minimal reproducible example (dataset shape, IR, the exact call that
  triggers the error, and the full traceback).

---

## Scope of contributions

CIPA is a research artifact accompanying a published paper. Breaking changes
to the seven complexity dimensions (D1–D7), their default weights, or the
Difficulty Score formula require discussion with the authors, as they affect
reproducibility of published results.

Contributions most likely to be accepted:
- Bug fixes in the implementation of existing dimensions.
- Performance improvements (faster MST, better subsampling strategies).
- Additional test coverage.
- Documentation improvements.
- New dataset loaders in `experiments/`.

---

## Adding a new dataset

See [experiments/ADDING_DATASETS.md](experiments/ADDING_DATASETS.md) for a
step-by-step guide covering the loader function, registration in `LOADERS` and
`TIER`, optional ground-truth values, and the verification checklist.
