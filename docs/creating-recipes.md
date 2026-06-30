# Creating Recipes

A recipe is a YAML file that defines what to download and how to validate it.
Recipes are stored in `recipes/` and are the single input to the full pipeline.

---

## Quick reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--create-recipe` | `wind` \| `currents` \| `waves` | — | **Required.** Which variable to validate |
| `--min-lon` | float (°) | `-20.0` | Western bound |
| `--max-lon` | float (°) | `0.0` | Eastern bound |
| `--min-lat` | float (°) | `35.0` | Southern bound |
| `--max-lat` | float (°) | `60.0` | Northern bound |
| `--start` | ISO-8601 date | `2026-01-01` | Start of time window (inclusive) |
| `--end` | ISO-8601 date | `2026-01-02` | End of time window (inclusive) |
| `--recipe-name` | string | `<type>_validation` | Custom name → sets the `name` field and the output filename |
| `--limit` | integer | none | Cap the number of SAR files downloaded |

---

## Examples

### Minimal — use all default

```bash
sar-validate --create-recipe wind
# → recipes/wind_validation.yaml
```

### Overide geographic bounds only
```bash
sar-validate --create-recipe wind \
  --min-lon -10 --max-lon 5 \
  --min-lat 50 --max-lat 65
# → recipes/wind_validation.yaml
```

### Override time window only
```bash
sar-validate --create-recipe currents \
  --start 2026-03-01 --end 2026-03-31

# → recipes/currents_validation.yaml
```

```bash
### Full override with a custom name
sar-validate --create-recipe wind \
  --min-lon -10 --max-lon 5 \
  --min-lat 50 --max-lat 65 \
  --start 2026-03-01 --end 2026-03-31 \
  --recipe-name north_sea_march_2026
# → recipes/north_sea_march_2026.yaml
```

### Limit number of SAR downloads
```bash
sar-validate --create-recipe waves \
  --start 2026-06-01 --end 2026-06-07 \
  --limit 5
# → recipes/waves_validation.yaml  (max_downloads: 5)
```

## Next steps after creating a recipe
# Check what will be downloaded without downloading anything
```bash
sar-validate --recipe recipes/wind_validation.yaml --dry-run
```

# Run the full download
```bash
sar-validate --recipe recipes/wind_validation.yaml
```