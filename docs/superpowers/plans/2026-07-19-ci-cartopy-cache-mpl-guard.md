# CI Chores: Cartopy Cache + Matplotlib Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CI fast and future-proof: cache the Natural Earth coastline data that currently dominates test-job runtime, silence the ~3000 cartopy deprecation warnings in CI logs, and guard CI against the Matplotlib 3.13 release that will turn those warnings into breakage.

**Architecture:** Three small, independent edits: (1) a targeted `filterwarnings` ignore in `pyproject.toml`'s pytest config; (2) an `actions/cache` step + pre-fetch step in `.github/workflows/ci.yml` for `~/.local/share/cartopy`; (3) a `matplotlib<3.13` constraint on the CI install line only (package metadata untouched). Final verification happens on the PR's own CI run.

**Tech Stack:** GitHub Actions (`actions/cache@v4`), pytest `filterwarnings`, cartopy 0.25 (`cartopy.io.shapereader.natural_earth` downloader). Spec: `docs/superpowers/specs/2026-07-19-rvl-swath-merge-and-ci-chores-design.md`.

## Global Constraints

- Run everything with the project venv: `.venv/bin/pytest`, `.venv/bin/python`.
- `ruff check .` must pass; baseline 414 tests passing, 0 warnings locally (local matplotlib is 3.10.9, so the deprecation never fires locally — the filter must be inert there, not error).
- Do NOT add the matplotlib pin to `pyproject.toml` dependencies/extras — CI install line only.
- The exact CI warning (captured from run 29682083994, job `test`, 3082 occurrences, all from `tests/test_visualization.py`):
  `/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/site-packages/cartopy/mpl/ticker.py:151: MatplotlibDeprecationWarning: The locs attribute was deprecated in Matplotlib 3.11 and will be removed in 3.13.`
- The CI test job needs exactly the two Natural Earth 10m physical datasets used by `_land_coastline_features` in `sar_validation/core/visualization.py`: `land` and `coastline`.
- cartopy's data dir on the Ubuntu runner is `~/.local/share/cartopy` (XDG default, confirmed with `cartopy.config["data_dir"]` on cartopy 0.25).

---

### Task 1: Targeted pytest ignore for the cartopy tick-formatter deprecation

**Files:**
- Modify: `pyproject.toml` (section `[tool.pytest.ini_options]`, currently just `testpaths = ["tests"]` at line 48-49)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: pytest config consumed by the CI run in Task 2's verification; no code interface.

- [ ] **Step 1: Add the filter**

In `pyproject.toml`, extend `[tool.pytest.ini_options]`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
filterwarnings = [
    # cartopy 0.25's tick formatter reads Matplotlib's Formatter.locs,
    # deprecated in mpl 3.11 and removed in 3.13 (~3000 hits per CI run on
    # mpl 3.11; never fires on mpl <3.11). Remove this ignore together with
    # the matplotlib<3.13 pin in .github/workflows/ci.yml once cartopy
    # ships a fix.
    "ignore:The locs attribute was deprecated in Matplotlib 3.11:matplotlib.MatplotlibDeprecationWarning",
]
```

Notes for the implementer: pytest `filterwarnings` entries are `action:message-regex:category`. The message part is a regex matched against the start of the warning text — the literal prefix above contains no regex metacharacters, so it can be used verbatim. `matplotlib.MatplotlibDeprecationWarning` is importable on every matplotlib ≥3.0, so the entry is valid (inert, not erroring) on the local 3.10.9 install.

- [ ] **Step 2: Verify the filter parses and the suite is unaffected locally**

Run: `.venv/bin/pytest`
Expected: 414 passed (415 if the RVL swath-merge PR is already merged), 0 warnings, and no `pytest.PytestConfigWarning`/import error about the filter entry. An unimportable category or malformed entry fails at session start — that's the failure mode this step catches.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "ci: ignore cartopy's mpl-3.11 'locs' deprecation warning in pytest"
```

---

### Task 2: Cache Natural Earth data and pin matplotlib in CI

**Files:**
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: nothing from Task 1 (independent edit; same PR).
- Produces: the final workflow verified on the PR's CI run.

- [ ] **Step 1: Rewrite the workflow test job**

Replace the `steps:` of the `test` job in `.github/workflows/ci.yml` with:

```yaml
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - name: Cache Natural Earth data
        uses: actions/cache@v4
        with:
          path: ~/.local/share/cartopy
          key: cartopy-natural-earth-10m-v1
      - name: Install package with dev extras
        # matplotlib pinned <3.13: cartopy 0.25's tick formatter uses
        # Formatter.locs, deprecated in mpl 3.11 and removed in 3.13.
        # Remove the pin (and the matching filterwarnings ignore in
        # pyproject.toml) once cartopy ships a fix.
        run: pip install -e .[dev] "matplotlib<3.13"
      - name: Pre-fetch Natural Earth coastline data
        # The 10m land + coastline shapefiles _land_coastline_features()
        # needs; instant no-op when the cache above was hit. Keeping this a
        # dedicated step (rather than letting pytest download lazily) makes
        # cold-cache download time visible in the job timeline.
        run: |
          python -c "
          import cartopy.io.shapereader as shpreader
          for name in ('land', 'coastline'):
              shpreader.natural_earth(resolution='10m', category='physical', name=name)
          "
      - name: Lint
        run: ruff check .
      - name: Test
        run: pytest
```

(The existing `checkout`/`setup-python`/`Lint`/`Test` steps are unchanged; the cache step, the pin on the install line, and the pre-fetch step are the additions. Step order matters: cache restore before pre-fetch, pre-fetch after install since it imports cartopy.)

- [ ] **Step 2: Sanity-check the YAML and the pre-fetch snippet locally**

```bash
.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('yaml ok')"
.venv/bin/python -c "
import cartopy.io.shapereader as shpreader
for name in ('land', 'coastline'):
    print(shpreader.natural_earth(resolution='10m', category='physical', name=name))
"
```

Expected: `yaml ok`, then two shapefile paths under `~/.local/share/cartopy/shapefiles/natural_earth/physical/` (instant, since the local machine already has them — proving the warm-cache no-op behavior).

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: cache Natural Earth data, pin matplotlib<3.13 until cartopy fix"
```

---

### Task 3: Verify on a real CI run

**Files:** none (verification only; runs after the branch is pushed and the PR opened by the finishing workflow).

**Interfaces:**
- Consumes: the pushed branch with Tasks 1-2.
- Produces: evidence for the PR (runtimes, clean warning summary).

- [ ] **Step 1: Watch the first (cold-cache) run**

```bash
gh pr checks --watch
```

Expected: green. Cold cache — the "Pre-fetch Natural Earth coastline data" step carries the download time that previously sat inside `pytest`.

- [ ] **Step 2: Inspect the run log**

```bash
gh run list --limit 1
gh api repos/LottevanH/sar-l2-validation-toolbox/actions/runs/<RUN_ID>/logs > /tmp/ci_logs.zip
unzip -o -q /tmp/ci_logs.zip -d /tmp/ci_logs
grep -c "The locs attribute was deprecated" /tmp/ci_logs/0_test.txt || echo "0 - warning gone"
grep -n "MatplotlibDeprecationWarning\|warnings summary" /tmp/ci_logs/0_test.txt | head
```

Expected: zero occurrences of the `locs` warning (previously 3082); pytest's warnings summary no longer lists `cartopy/mpl/ticker.py`. Also confirm from `gh run view <RUN_ID>` that matplotlib installed is <3.13 (visible in the pip install log if needed: `grep -i "matplotlib" /tmp/ci_logs/0_test.txt | head`).

- [ ] **Step 3: Re-run to verify the warm cache**

```bash
gh run rerun <RUN_ID>
gh run watch <NEW_OR_SAME_RUN_ID>
```

Expected: "Cache Natural Earth data" reports a cache hit, the pre-fetch step completes in ~a second, and the `Test` step is markedly faster than the pre-cache baseline (baseline job total: ~2m20s, dominated by the in-test download). Record the before/after job durations in the PR description.

---

## Self-review checklist (run after writing code)

- Spec coverage: warnings filter (Task 1), Natural Earth cache + pre-fetch (Task 2), CI-only matplotlib pin with removal condition (Task 2), runtime/log verification (Task 3). All three spec items covered; nothing in this plan touches package metadata.
