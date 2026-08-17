# Contributing

Thank you for contributing to the SAR Level 2 validation toolbox.

This project is developed in a collaborative research environment, so we value clear communication, reproducible changes, and tests that document expected behavior.

## Community expectations

- Be respectful and constructive in issues, reviews, and pull requests.
- Keep discussions technical and focused on the project goals.
- Prefer small, reviewable changes over broad refactors.
- If a change affects scientific behavior, include a brief explanation of the method, assumptions, and validation performed.

## Before you start

1. Open or check an issue for the work you plan to do when the change is non-trivial.
2. Confirm the current branch strategy and target branch in the repository.
3. Make sure you have a working Python environment that matches the project configuration.

## Development setup

Clone the repository and install the project in editable mode with the development extras:

```bash
git clone <repository-url>
cd qa4sar
python -m pip install -U pip
python -m pip install -e '.[dev]'
```

For the heavier soil-moisture workflow, install the optional extra as needed:

```bash
python -m pip install -e '.[soil_moisture]'
```

The project uses:

- `pytest` for tests
- `ruff` for linting
- `mypy` for static type checks
- `src/` layout packaging via `setuptools`

## Branching and workflow

Create a feature branch from the active integration branch before making changes:

```bash
git checkout -b feature/my-change
```

Use concise, descriptive branch names such as:

- `feature/era5-validation`
- `fix/soil-moisture-imports`
- `docs/contributing-guide`

Keep commits focused and include meaningful messages. If a fix addresses an issue, reference it in the commit or PR description.

## Code and documentation standards

- Follow the lint and formatting rules defined in `pyproject.toml`.
- Keep imports organized and avoid introducing unused dependencies.
- Prefer clear, explicit names over terse abbreviations where possible.
- Add or update tests when changing behavior, fixing bugs, or adding features.
- Update documentation when user-facing behavior changes or new configuration is introduced.

## Testing

Run the relevant tests locally before opening a pull request:

```bash
pytest -q
```

For a narrower validation loop while developing, run only the affected test subset:

```bash
pytest tests/<specific_test_file>.py -q
```

If you change packaging or module layout, confirm the editable install still works:

```bash
python -m pip install -e .
```

## Pull requests

Before opening a pull request:

- Rebase or update your branch with the target branch if needed.
- Ensure tests pass locally.
- Check that linting and typing checks are clean.
- Summarize the rationale, scientific assumptions, and validation in the PR description.

A good PR description includes:

- a short summary of the change,
- the motivation or issue addressed,
- validation commands run,
- any known limitations or follow-up items.

## Reporting issues

Use the issue templates in `.github/ISSUE_TEMPLATE/` when available. Include enough context to reproduce the problem, such as:

- Python version,
- platform,
- command or script used,
- expected vs actual behavior,
- any relevant error logs.

## Questions

If you are unsure about the right place for a change, start with a discussion in an issue or open a draft pull request to clarify the approach.
