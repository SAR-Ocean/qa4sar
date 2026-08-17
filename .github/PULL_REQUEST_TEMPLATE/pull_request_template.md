# Pull Request

Describe the change and the motivation. Link relevant issues and include testing notes.

## Checklist

- [ ] Changes are described and motivated
- [ ] Tests added / updated
- [ ] Documentation updated (README, CHANGELOG, RELEASE.md as needed)
- [ ] Linting and type checks pass (`ruff`, `mypy`)
- [ ] DCO sign-off on commits (see DCO.md)
- [ ] Security implications considered (see SECURITY.md)
- [ ] If this is a release, `CHANGELOG.md` and `RELEASE.md` updated
- [ ] Assign reviewers or rely on `CODEOWNERS` for automatic review

If your change adds a new dependency or changes packaging, ensure
`pyproject.toml` is updated and an editable install still works:

```bash
python -m pip install -e .
```

Add any additional notes for reviewers below.
