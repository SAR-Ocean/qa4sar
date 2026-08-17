# Release Process

This document describes the recommended release process for the project.

1. Update `CHANGELOG.md` with a summary of changes under an Unreleased heading.
2. Bump the version in `pyproject.toml` (follow semver: MAJOR.MINOR.PATCH).
3. Run the test suite and linting locally:

```bash
python -m pip install -e '.[dev]'
pytest
ruff check .
python -m mypy -p sar_validation
```

4. Commit the version and changelog updates with a signed commit:

```bash
git commit -s -am "Bump version to X.Y.Z"
```

5. Tag the release and push tags:

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin --tags
```

6. Create a GitHub Release from the tag and paste the changelog notes.

7. If necessary, publish the package to PyPI following your org's release policy.

Note: update any deployment or packaging automation (CI/CD) that needs the new version.
