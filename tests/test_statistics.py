"""Tests for sar_validation.core.statistics."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sar_validation.core._variable_map import filter_variable_pairs, infer_variable_pairs
from sar_validation.core.recipe import Recipe, RecipeConfig, SARDataSpec
from sar_validation.core.statistics import (
    _assemble_stats_dataset,
    _core_metrics,
    _group_by_columns,
    _missing_columns,
    add_rescaled_sar_column,
    compute_statistics,
    run_statistics,
    save_statistics,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def collocation_ds():
    """Synthetic collocation dataset with two sources."""
    n = 40
    rng = np.random.default_rng(42)
    sar_vals = rng.uniform(2, 14, size=n)
    # Correlated validation values with small noise
    val_vals = sar_vals + rng.normal(0, 0.5, size=n)

    sources = ["mooring"] * 20 + ["buoy"] * 20

    ds = xr.Dataset(
        {
            "sar_owiWindSpeed": ("collocation", sar_vals),
            "val_WSPD":         ("collocation", val_vals),
            "val_source":       ("collocation", sources),
            "sar_lon":          ("collocation", rng.uniform(-10, 5, n)),
            "sar_lat":          ("collocation", rng.uniform(50, 65, n)),
            "val_lon":          ("collocation", rng.uniform(-10, 5, n)),
            "val_lat":          ("collocation", rng.uniform(50, 65, n)),
        }
    )
    return ds


# ---------------------------------------------------------------------------
# compute_statistics
# ---------------------------------------------------------------------------

class TestComputeStatistics:
    def test_basic_properties(self, collocation_ds):
        ds = compute_statistics(collocation_ds, "owiWindSpeed", "WSPD")

        assert isinstance(ds, xr.Dataset)
        for metric in ("N", "bias", "std", "rmse", "correlation", "scatter_index"):
            assert metric in ds.data_vars, f"Missing metric: {metric}"
        assert "source" in ds.dims
        assert set(ds["source"].values) == {"mooring", "buoy"}
        # Synthetic data has small noise so bias should be small
        assert abs(float(ds["bias"].mean())) < 1.0
        assert (ds["rmse"].values >= 0).all()
        corr = ds["correlation"].values
        assert np.all((corr >= -1) & (corr <= 1))
        total_n = int(ds["N"].sum())
        assert total_n == 40

    def test_missing_var_returns_none(self, collocation_ds):
        result = compute_statistics(collocation_ds, "owiWaveHeight", "VHM0")
        assert result is None


# ---------------------------------------------------------------------------
# compute_statistics_soil_moisture (pytesmo CDF-matching + ubRMSD)
# ---------------------------------------------------------------------------

@pytest.fixture
def soil_moisture_collocation_ds():
    """Synthetic soil-moisture collocation dataset, single source."""
    n = 30
    rng = np.random.default_rng(7)
    val_vals = rng.uniform(0.05, 0.35, size=n)          # ISMN volumetric fraction
    sar_vals = val_vals * 150.0 + rng.normal(0, 2, n)    # SAR in a different domain (% saturation-like)

    return xr.Dataset({
        "sar_sarSSM":        ("collocation", sar_vals),
        "val_SOIL_MOISTURE": ("collocation", val_vals),
        "val_source":        ("collocation", ["ismn"] * n),
    })


class _FakeCdfMatchScale:
    """Stand-in for pytesmo.scaling.scale(method='cdf_match', reference_index=1).

    Rescales column 0 (SAR) linearly onto column 1 (val)'s mean/std — good
    enough to exercise the statistics-module wiring without depending on
    pytesmo's real CDF-matching implementation.
    """

    def __call__(self, df, method="cdf_match", reference_index=1, **kwargs):
        # Real pytesmo.scaling.scale holds the column at `reference_index`
        # fixed and rescales every other column onto it; for this 2-column
        # (sar, val) case, that's whichever column isn't the reference.
        out = df.copy()
        ref_col = df.columns[reference_index]
        src_col = df.columns[1 - reference_index]
        src = df[src_col].values
        ref = df[ref_col].values
        rescaled = (src - src.mean()) / src.std() * ref.std() + ref.mean()
        out[src_col] = rescaled
        return out


class _FakeCdfMatching:
    """Stand-in for pytesmo.cdf_matching.CDFMatching.

    Stores reference (val) data and rescales arbitrary input data onto it
    using the same linear mean/std approach as _FakeCdfMatchScale.
    """

    def __init__(self, nbins=10, minobs=20):
        self.nbins = nbins
        self.minobs = minobs
        self.ref_mean = None
        self.ref_std = None
        self.src_mean = None
        self.src_std = None

    def fit(self, src_vals, ref_vals):
        """Store reference and source statistics for later prediction."""
        self.ref_mean = np.mean(ref_vals)
        self.ref_std = np.std(ref_vals)
        self.src_mean = np.mean(src_vals)
        self.src_std = np.std(src_vals)

    def predict(self, src_vals):
        """Rescale input values onto the reference's domain."""
        if self.src_std == 0 or self.ref_std == 0:
            return src_vals.copy()
        return (src_vals - self.src_mean) / self.src_std * self.ref_std + self.ref_mean


def _fake_ubrmsd(sar_vals, val_vals):
    diff = sar_vals - sar_vals.mean() - (val_vals - val_vals.mean())
    return float(np.sqrt(np.mean(diff ** 2)))


def _patched_pytesmo_modules():
    fake_scaling = MagicMock()
    fake_scaling.scale = _FakeCdfMatchScale()
    fake_metrics = MagicMock()
    fake_metrics.ubrmsd = _fake_ubrmsd
    fake_cdf_matching = MagicMock()
    fake_cdf_matching.CDFMatching = _FakeCdfMatching
    fake_pytesmo = MagicMock()
    return {
        "pytesmo": fake_pytesmo,
        "pytesmo.scaling": fake_scaling,
        "pytesmo.metrics": fake_metrics,
        "pytesmo.cdf_matching": fake_cdf_matching,
    }


class TestComputeStatisticsSoilMoisture:
    def test_returns_dataset_with_ubrmsd(self, soil_moisture_collocation_ds):
        from sar_validation.core.statistics import compute_statistics_soil_moisture

        with patch.dict("sys.modules", _patched_pytesmo_modules()):
            ds = compute_statistics_soil_moisture(
                soil_moisture_collocation_ds, "sarSSM", "SOIL_MOISTURE",
            )

        assert ds is not None
        for metric in ("N", "bias", "std", "rmse", "correlation", "scatter_index", "ubrmsd"):
            assert metric in ds.data_vars, f"Missing metric: {metric}"

    def test_correlation_high_after_rescaling(self, soil_moisture_collocation_ds):
        from sar_validation.core.statistics import compute_statistics_soil_moisture

        with patch.dict("sys.modules", _patched_pytesmo_modules()):
            ds = compute_statistics_soil_moisture(
                soil_moisture_collocation_ds, "sarSSM", "SOIL_MOISTURE",
            )
        # SAR was built as a noisy linear function of val, so correlation
        # should be strong even before considering the rescaling.
        assert float(ds["correlation"].values[0]) > 0.9

    def test_missing_column_returns_none(self, soil_moisture_collocation_ds):
        from sar_validation.core.statistics import compute_statistics_soil_moisture

        with patch.dict("sys.modules", _patched_pytesmo_modules()):
            ds = compute_statistics_soil_moisture(
                soil_moisture_collocation_ds, "sarSSM", "NOT_A_REAL_VAR",
            )
        assert ds is None

    def test_degenerate_cdf_match_skips_group_instead_of_reporting_nan(
        self, soil_moisture_collocation_ds, caplog,
    ):
        """Real pytesmo can silently CDF-match to an all-NaN series when its
        percentile binning degenerates for a small/coarsely-quantized sample
        (observed against a real collocation run) — this must be skipped
        with a warning, not surfaced as a row of NaN metrics that looks like
        a normally-computed (near-)zero-signal result.
        """
        from sar_validation.core.statistics import compute_statistics_soil_moisture

        fake_scaling = MagicMock()
        fake_scaling.scale = MagicMock(
            side_effect=lambda df, **kwargs: df.assign(**{df.columns[0]: np.nan})
        )
        fake_metrics = MagicMock()
        fake_metrics.ubrmsd = _fake_ubrmsd
        modules = {"pytesmo": MagicMock(), "pytesmo.scaling": fake_scaling, "pytesmo.metrics": fake_metrics}

        with patch.dict("sys.modules", modules):
            ds = compute_statistics_soil_moisture(
                soil_moisture_collocation_ds, "sarSSM", "SOIL_MOISTURE",
            )
        assert ds is None
        assert "degenerated to all-NaN" in caplog.text

    def test_skips_groups_with_fewer_than_2_pairs(self):
        """Test that groups with <2 valid pairs are skipped, not included with garbage values.

        The CDF-matching operation requires at least 2 points. Groups with only 1 pair
        should be silently excluded (with a warning) rather than crashing or producing
        invalid output.
        """
        from sar_validation.core.statistics import compute_statistics_soil_moisture

        rng = np.random.default_rng(42)

        # Create a dataset with two groups:
        # - "ismn_many": 5 valid pairs (should be included)
        # - "ismn_few": 1 valid pair (should be skipped)
        val_vals_many = rng.uniform(0.05, 0.35, size=5)
        sar_vals_many = val_vals_many * 150.0 + rng.normal(0, 2, 5)

        val_vals_few = np.array([0.15])
        sar_vals_few = np.array([22.5])

        val_vals = np.concatenate([val_vals_many, val_vals_few])
        sar_vals = np.concatenate([sar_vals_many, sar_vals_few])
        sources = ["ismn_many"] * 5 + ["ismn_few"] * 1

        ds = xr.Dataset({
            "sar_sarSSM":        ("collocation", sar_vals),
            "val_SOIL_MOISTURE": ("collocation", val_vals),
            "val_source":        ("collocation", sources),
        })

        with patch.dict("sys.modules", _patched_pytesmo_modules()):
            result = compute_statistics_soil_moisture(ds, "sarSSM", "SOIL_MOISTURE")

        # The result should not be None (ismn_many has >= 2 pairs)
        assert result is not None

        # The source coordinate should only contain the larger group
        source_labels = set(result["source"].values)
        assert "ismn_many" in source_labels, "Large group (5 pairs) should be included"
        assert "ismn_few" not in source_labels, "Small group (1 pair) should be skipped"

        # Verify that the included group has the correct count
        assert int(result["N"].values[0]) == 5, "Included group should have N=5"

    def test_ascat_row_reports_volumetric_domain_stats(self):
        """The ASCAT row in the CDF-matched statistics table must reflect
        volumetric-domain bias/rmse/ubrmsd (small numbers, ~0-1 scale), not
        percent-domain ones (~0-100 scale) -- consistent with what the
        CDF-matched plots now show (Task 2)."""
        from sar_validation.core.statistics import compute_statistics_soil_moisture

        rng = np.random.default_rng(31)
        n_ismn, n_ascat = 15, 15
        ismn_val = rng.uniform(0.05, 0.35, n_ismn)
        ismn_sar = ismn_val * 150.0 + rng.normal(0, 1, n_ismn)
        ascat_val = rng.uniform(10.0, 90.0, n_ascat)
        ascat_sar = ascat_val + rng.normal(0, 1, n_ascat)

        ds = xr.Dataset({
            "sar_sarSSM": xr.DataArray(
                np.concatenate([ismn_sar, ascat_sar]), dims="collocation", attrs={"units": "%"},
            ),
            "val_SOIL_MOISTURE": xr.DataArray(
                np.concatenate([ismn_val, ascat_val]), dims="collocation",
                attrs={"units": "mixed — see val_units"},
            ),
            "val_source": ("collocation", ["ismn"] * n_ismn + ["ascat_ssm"] * n_ascat),
            "val_units": ("collocation", np.array(["1"] * n_ismn + ["%"] * n_ascat)),
        })

        with patch.dict("sys.modules", _patched_pytesmo_modules()):
            stats_ds = compute_statistics_soil_moisture(ds, "sarSSM", "SOIL_MOISTURE")

        assert stats_ds is not None
        ascat_idx = list(stats_ds["source"].values).index("ascat_ssm")
        assert abs(float(stats_ds["bias"].values[ascat_idx])) < 1.0
        assert float(stats_ds["rmse"].values[ascat_idx]) < 1.0

    def test_ascat_row_absent_when_ismn_missing(self):
        """No reference source to convert onto -- ASCAT must be dropped
        from the CDF-matched statistics table entirely for this run, not
        reported with (wrong) percent-domain numbers."""
        from sar_validation.core.statistics import compute_statistics_soil_moisture

        rng = np.random.default_rng(32)
        n = 10
        ascat_val = rng.uniform(10.0, 90.0, n)
        ascat_sar = ascat_val + rng.normal(0, 1, n)

        ds = xr.Dataset({
            "sar_sarSSM": xr.DataArray(ascat_sar, dims="collocation", attrs={"units": "%"}),
            "val_SOIL_MOISTURE": ("collocation", ascat_val),
            "val_source": ("collocation", ["ascat_ssm"] * n),
        })

        with patch.dict("sys.modules", _patched_pytesmo_modules()):
            stats_ds = compute_statistics_soil_moisture(ds, "sarSSM", "SOIL_MOISTURE")

        assert stats_ds is None


class TestAddRescaledSarColumn:
    def test_rescaled_sar_lands_in_val_domain(self, soil_moisture_collocation_ds):
        from sar_validation.core.statistics import add_rescaled_sar_column

        val_vals = soil_moisture_collocation_ds["val_SOIL_MOISTURE"].values

        with patch.dict("sys.modules", _patched_pytesmo_modules()):
            out = add_rescaled_sar_column(
                soil_moisture_collocation_ds, "sarSSM", "SOIL_MOISTURE",
            )

        rescaled = out["sar_sarSSM"].values
        assert not np.any(np.isnan(rescaled))
        # The fake CDF-match linearly matches mean/std onto val's — a much
        # tighter check than "just don't crash": confirms the values
        # actually moved into val's numeric domain, not just no NaNs.
        assert rescaled.mean() == pytest.approx(val_vals.mean(), abs=1e-6)
        assert rescaled.std() == pytest.approx(val_vals.std(), abs=1e-6)

    def test_does_not_mutate_original_dataset(self, soil_moisture_collocation_ds):
        from sar_validation.core.statistics import add_rescaled_sar_column

        original_sar = soil_moisture_collocation_ds["sar_sarSSM"].values.copy()

        with patch.dict("sys.modules", _patched_pytesmo_modules()):
            add_rescaled_sar_column(soil_moisture_collocation_ds, "sarSSM", "SOIL_MOISTURE")

        np.testing.assert_array_equal(
            soil_moisture_collocation_ds["sar_sarSSM"].values, original_sar,
        )

    def test_rescaled_column_attrs_copied_from_val(self):
        import xarray as xr

        from sar_validation.core.statistics import add_rescaled_sar_column

        ds = xr.Dataset({
            "sar_sarSSM": ("collocation", np.array([50.0, 60.0, 55.0], dtype=float)),
            "val_SOIL_MOISTURE": xr.DataArray(
                np.array([0.2, 0.25, 0.22]), dims="collocation",
                attrs={"units": "1", "long_name": "ISMN soil moisture"},
            ),
            "val_source": ("collocation", ["ismn"] * 3),
        })

        with patch.dict("sys.modules", _patched_pytesmo_modules()):
            out = add_rescaled_sar_column(ds, "sarSSM", "SOIL_MOISTURE")

        assert out["sar_sarSSM"].attrs["units"] == "1"
        assert out["sar_sarSSM"].attrs["long_name"] == "ISMN soil moisture"

    def test_missing_column_returns_unchanged_copy(self, soil_moisture_collocation_ds, caplog):
        from sar_validation.core.statistics import add_rescaled_sar_column

        with patch.dict("sys.modules", _patched_pytesmo_modules()):
            out = add_rescaled_sar_column(soil_moisture_collocation_ds, "sarSSM", "NOT_A_REAL_VAR")

        np.testing.assert_array_equal(
            out["sar_sarSSM"].values, soil_moisture_collocation_ds["sar_sarSSM"].values,
        )
        assert "not found" in caplog.text

    def test_small_group_left_as_nan(self):
        import xarray as xr

        from sar_validation.core.statistics import add_rescaled_sar_column

        ds = xr.Dataset({
            "sar_sarSSM": ("collocation", np.array([50.0, 60.0, 55.0, 58.0, 52.0, 90.0])),
            "val_SOIL_MOISTURE": ("collocation", np.array([0.20, 0.25, 0.22, 0.24, 0.21, 0.30])),
            "val_source": ("collocation", ["ismn_many"] * 5 + ["ismn_one"]),
        })

        with patch.dict("sys.modules", _patched_pytesmo_modules()):
            out = add_rescaled_sar_column(ds, "sarSSM", "SOIL_MOISTURE")

        rescaled = out["sar_sarSSM"].values
        assert not np.any(np.isnan(rescaled[:5])), "5-pair group should be rescaled"
        assert np.isnan(rescaled[5]), "1-pair group can't CDF-match — left as NaN"

    def test_ascat_group_converted_to_volumetric_and_not_double_matched(self):
        """ASCAT's group must be harmonized into ISMN's domain (Task 1),
        not additionally re-matched by add_rescaled_sar_column's own
        per-group _cdf_match_sar_series loop -- that would double-apply
        CDF-matching and corrupt the values."""
        rng = np.random.default_rng(21)
        n_ismn, n_ascat = 15, 15
        ismn_val = rng.uniform(0.05, 0.35, n_ismn)
        ismn_sar = ismn_val * 150.0 + rng.normal(0, 1, n_ismn)
        ascat_val = rng.uniform(10.0, 90.0, n_ascat)
        ascat_sar = ascat_val + rng.normal(0, 1, n_ascat)

        ds = xr.Dataset({
            "sar_sarSSM": xr.DataArray(
                np.concatenate([ismn_sar, ascat_sar]), dims="collocation", attrs={"units": "%"},
            ),
            "val_SOIL_MOISTURE": xr.DataArray(
                np.concatenate([ismn_val, ascat_val]), dims="collocation",
                attrs={"units": "mixed — see val_units"},
            ),
            "val_source": ("collocation", ["ismn"] * n_ismn + ["ascat_ssm"] * n_ascat),
            "val_units": ("collocation", np.array(["1"] * n_ismn + ["%"] * n_ascat)),
        })

        with patch.dict("sys.modules", _patched_pytesmo_modules()):
            out = add_rescaled_sar_column(ds, "sarSSM", "SOIL_MOISTURE")

        is_ascat = np.array(out["val_source"].values) == "ascat_ssm"
        assert np.nanmax(out["sar_sarSSM"].values[is_ascat]) < 1.0
        assert not np.any(np.isnan(out["sar_sarSSM"].values[is_ascat]))

    def test_ascat_group_dropped_when_ismn_absent(self, caplog):
        """When ISMN isn't present to define the reference transform,
        ASCAT's rows must come back NaN, not silently left in the percent
        domain (which would reintroduce the original mixed-domain bug)."""
        rng = np.random.default_rng(22)
        n = 10
        ascat_val = rng.uniform(10.0, 90.0, n)
        ascat_sar = ascat_val + rng.normal(0, 1, n)

        ds = xr.Dataset({
            "sar_sarSSM": xr.DataArray(ascat_sar, dims="collocation", attrs={"units": "%"}),
            "val_SOIL_MOISTURE": ("collocation", ascat_val),
            "val_source": ("collocation", ["ascat_ssm"] * n),
        })

        with patch.dict("sys.modules", _patched_pytesmo_modules()):
            out = add_rescaled_sar_column(ds, "sarSSM", "SOIL_MOISTURE")

        assert np.all(np.isnan(out["sar_sarSSM"].values))


class TestAddRescaledSarColumnInheritsHonestMixedUnitsLabel:
    def test_rescaled_column_gets_mixed_marker_when_val_col_is_mixed(self):
        """add_rescaled_sar_column does `out[sar_col].attrs =
        dict(out[val_col].attrs)` -- once annotate_collocation_ds (see
        test_cf_metadata.py) stops lying about val_SOIL_MOISTURE's units
        for a mixed-source run, this blind copy automatically becomes
        correct too, with no source change needed here. This test pins
        that down as an explicit regression guard."""
        import numpy as np
        import xarray as xr

        from sar_validation.core.statistics import add_rescaled_sar_column

        n = 40
        rng = np.random.default_rng(0)
        collocation_ds = xr.Dataset({
            "sar_sarSSM": ("collocation", rng.uniform(0, 100, n)),
            "val_SOIL_MOISTURE": ("collocation", rng.uniform(0, 100, n)),
            "val_source": ("collocation", ["ascat_ssm"] * n),
        })
        collocation_ds["val_SOIL_MOISTURE"].attrs["units"] = "mixed — see val_units"

        out = add_rescaled_sar_column(collocation_ds, "sarSSM", "SOIL_MOISTURE")

        assert out["sar_sarSSM"].attrs.get("units") == "mixed — see val_units"


class TestCdfMatchingBinsResizedWarningSuppressed:
    """Real (unmocked) pytesmo.cdf_matching resizes its bins and emits
    'UserWarning: The bins have been resized' whenever nsamples * (100 /
    nbins) / 100 < minobs=20 -- an expected, routine side effect of our own
    deliberately small nbins cap (see _cdf_match_sar_series), not a sign of
    a problem. Lotte saw this leak straight to a real CLI run's console;
    it must not propagate to callers of either public entry point."""

    def test_add_rescaled_sar_column_suppresses_it(self, soil_moisture_collocation_ds):
        from sar_validation.core.statistics import add_rescaled_sar_column

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            add_rescaled_sar_column(soil_moisture_collocation_ds, "sarSSM", "SOIL_MOISTURE")

        assert not any("bins have been resized" in str(w.message).lower() for w in caught)

    def test_fit_sar_to_val_transform_suppresses_it(self, soil_moisture_collocation_ds):
        from sar_validation.core.statistics import fit_sar_to_val_transform

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit_sar_to_val_transform(soil_moisture_collocation_ds, "sarSSM", "SOIL_MOISTURE")

        assert not any("bins have been resized" in str(w.message).lower() for w in caught)


class TestHarmonizePercentDomainSources:
    """_harmonize_percent_domain_sources converts a percent-domain
    val_source (e.g. ASCAT, "%") into the reference source's (ISMN's)
    volumetric domain, by reusing the CDF-match transform fit from
    SAR-vs-ISMN pairs -- since SAR's own raw retrieval shares ASCAT's "%"
    domain (see design-choices.md SS8.7), that same transform is valid
    applied to ASCAT's raw values too."""

    def _mixed_ds(self, n_ismn=15, n_ascat=15, sar_units="%"):
        rng = np.random.default_rng(11)
        ismn_val = rng.uniform(0.05, 0.35, n_ismn)
        ismn_sar = ismn_val * 150.0 + rng.normal(0, 1, n_ismn)
        ascat_val = rng.uniform(10.0, 90.0, n_ascat)   # ASCAT's own "%" scale
        ascat_sar = ascat_val + rng.normal(0, 1, n_ascat)

        sar_vals = np.concatenate([ismn_sar, ascat_sar])
        val_vals = np.concatenate([ismn_val, ascat_val])
        sources = ["ismn"] * n_ismn + ["ascat_ssm"] * n_ascat

        return xr.Dataset({
            "sar_sarSSM": xr.DataArray(
                sar_vals, dims="collocation", attrs={"units": sar_units},
            ),
            "val_SOIL_MOISTURE": xr.DataArray(
                val_vals, dims="collocation", attrs={"units": "mixed — see val_units"},
            ),
            "val_source": ("collocation", sources),
            "val_units": (
                "collocation", np.array(["1"] * n_ismn + ["%"] * n_ascat),
            ),
            "val_long_name": (
                "collocation",
                np.array(["ISMN soil moisture"] * n_ismn + ["ASCAT soil moisture"] * n_ascat),
            ),
        })

    def test_ascat_values_move_into_ismns_numeric_range(self):
        from sar_validation.core.statistics import _harmonize_percent_domain_sources

        ds = self._mixed_ds()
        out, converted, dropped = _harmonize_percent_domain_sources(ds, "sarSSM", "SOIL_MOISTURE")

        assert converted == {"ascat_ssm"}
        assert dropped == set(), "successful conversion must not also report the source as dropped"
        is_ascat = np.array(out["val_source"].values) == "ascat_ssm"
        ascat_val_converted = out["val_SOIL_MOISTURE"].values[is_ascat]
        ascat_sar_converted = out["sar_sarSSM"].values[is_ascat]
        # Was ~10-90 (percent); must now land near ISMN's ~0.05-0.35 range.
        assert np.nanmax(ascat_val_converted) < 1.0
        assert np.nanmax(ascat_sar_converted) < 1.0
        # ISMN's own rows must be untouched.
        is_ismn = np.array(out["val_source"].values) == "ismn"
        np.testing.assert_array_equal(
            out["val_SOIL_MOISTURE"].values[is_ismn], ds["val_SOIL_MOISTURE"].values[is_ismn],
        )

    def test_val_units_companion_updated_for_converted_rows(self):
        from sar_validation.core.statistics import _harmonize_percent_domain_sources

        ds = self._mixed_ds()
        out, _converted, _dropped = _harmonize_percent_domain_sources(ds, "sarSSM", "SOIL_MOISTURE")

        is_ascat = np.array(out["val_source"].values) == "ascat_ssm"
        assert all(u == "1" for u in np.array(out["val_units"].values)[is_ascat])
        assert all(
            n == "ISMN soil moisture" for n in np.array(out["val_long_name"].values)[is_ascat]
        )
        # Every source now shares one family -- the column-level "mixed"
        # sentinel must collapse back to a real units string.
        assert out["val_SOIL_MOISTURE"].attrs["units"] == "1"

    @pytest.mark.parametrize(
        "make_ds,patch_target,expected_log_substr,check_sar_nan,check_ismn_untouched",
        [
            pytest.param(
                lambda self: self._mixed_ds(n_ismn=0, n_ascat=10).isel(collocation=slice(0, 10)),
                None,
                "absent",
                True,
                False,
                id="reference_absent",
            ),
            pytest.param(
                lambda self: self._mixed_ds(n_ismn=1, n_ascat=10),
                None,
                "< 2 valid",
                False,
                False,
                id="reference_too_sparse",
            ),
            pytest.param(
                lambda self: self._mixed_ds(n_ismn=15, n_ascat=10),
                "pytesmo.cdf_matching.CDFMatching.fit",
                "CDF-matching fit failed",
                True,
                True,
                id="cdf_matching_fit_raises",
            ),
        ],
    )
    def test_unavailable_reference_drops_percent_sources(
        self, make_ds, patch_target, expected_log_substr, check_sar_nan,
        check_ismn_untouched, caplog,
    ):
        """If the reference source (ISMN) is absent, too sparse, or its
        CDF-matching fit raises, _harmonize_percent_domain_sources must
        degrade gracefully: drop the to-be-converted rows to NaN, log a
        warning, and report them as dropped -- not propagate an exception
        or silently leave them in the wrong (percent) domain."""
        from sar_validation.core.statistics import _harmonize_percent_domain_sources

        ds = make_ds(self)

        if patch_target is not None:
            with patch(patch_target) as mock_fit:
                mock_fit.side_effect = ValueError("degenerate reference values")
                out, converted, dropped = _harmonize_percent_domain_sources(
                    ds, "sarSSM", "SOIL_MOISTURE",
                )
        else:
            out, converted, dropped = _harmonize_percent_domain_sources(ds, "sarSSM", "SOIL_MOISTURE")

        assert converted == set()
        assert dropped == {"ascat_ssm"}, (
            "ascat_ssm needed converting but couldn't -- must be "
            "reported as dropped, distinct from 'nothing needed converting'"
        )
        is_ascat = np.array(out["val_source"].values) == "ascat_ssm"
        assert np.all(np.isnan(out["val_SOIL_MOISTURE"].values[is_ascat]))
        if check_sar_nan:
            assert np.all(np.isnan(out["sar_sarSSM"].values[is_ascat]))
        if check_ismn_untouched:
            is_ismn = np.array(out["val_source"].values) == "ismn"
            np.testing.assert_array_equal(
                out["val_SOIL_MOISTURE"].values[is_ismn], ds["val_SOIL_MOISTURE"].values[is_ismn],
            )
        assert expected_log_substr in caplog.text

    def test_no_percent_domain_source_present_is_a_true_noop(self):
        """No ASCAT-like source present (e.g. a plain ISMN-only run, or any
        non-soil-moisture recipe) -- must return the exact same object, not
        even a copy, since every call site invokes this unconditionally."""
        from sar_validation.core.statistics import _harmonize_percent_domain_sources

        ds = xr.Dataset({
            "sar_sarSSM": xr.DataArray(
                np.array([10.0, 20.0, 30.0]), dims="collocation", attrs={"units": "%"},
            ),
            "val_SOIL_MOISTURE": ("collocation", np.array([0.1, 0.2, 0.3])),
            "val_source": ("collocation", ["ismn"] * 3),
        })

        out, converted, dropped = _harmonize_percent_domain_sources(ds, "sarSSM", "SOIL_MOISTURE")

        assert out is ds
        assert converted == set()
        assert dropped == set()

    def test_non_soil_moisture_dataset_is_a_true_noop(self):
        """A wind/wave/currents-shaped dataset (val_source labels never in
        _VAL_SOURCE_UNITS_FAMILY) must also short-circuit to a no-op."""
        from sar_validation.core.statistics import _harmonize_percent_domain_sources

        ds = xr.Dataset({
            "sar_owiWindSpeed": xr.DataArray(
                np.array([5.0, 6.0, 7.0]), dims="collocation", attrs={"units": "m s-1"},
            ),
            "val_WSPD": ("collocation", np.array([5.1, 6.1, 7.1])),
            "val_source": ("collocation", ["mooring"] * 3),
        })

        out, converted, dropped = _harmonize_percent_domain_sources(ds, "owiWindSpeed", "WSPD")

        assert out is ds
        assert converted == set()
        assert dropped == set()


class TestFitSarToValTransformHarmonizesFirst:
    def test_transform_maps_ascat_field_into_ismn_range_not_its_own(self):
        """Before this fix, fit_sar_to_val_transform pooled every
        val_source's raw pairs with no grouping -- a transform fit that way
        on mixed percent+volumetric data is nonsense. After harmonizing
        first, a raw-percent SAR field value should map into ISMN's
        volumetric range, not stay near its own raw percent value."""
        from sar_validation.core.statistics import fit_sar_to_val_transform

        rng = np.random.default_rng(41)
        n_ismn, n_ascat = 15, 15
        ismn_val = rng.uniform(0.05, 0.35, n_ismn)
        ismn_sar = ismn_val * 150.0 + rng.normal(0, 1, n_ismn)
        ascat_val = rng.uniform(10.0, 90.0, n_ascat)
        ascat_sar = ascat_val + rng.normal(0, 1, n_ascat)

        ds = xr.Dataset({
            "sar_sarSSM": xr.DataArray(
                np.concatenate([ismn_sar, ascat_sar]), dims="collocation", attrs={"units": "%"},
            ),
            "val_SOIL_MOISTURE": xr.DataArray(
                np.concatenate([ismn_val, ascat_val]), dims="collocation",
                attrs={"units": "mixed — see val_units"},
            ),
            "val_source": ("collocation", ["ismn"] * n_ismn + ["ascat_ssm"] * n_ascat),
            "val_units": ("collocation", np.array(["1"] * n_ismn + ["%"] * n_ascat)),
        })

        transform = fit_sar_to_val_transform(ds, "sarSSM", "SOIL_MOISTURE")
        assert transform is not None

        # A raw SAR field value in ASCAT's percent range (~50) must map
        # near ISMN's volumetric range (~0.05-0.35), not stay near 50.
        mapped = transform(np.array([50.0]))
        assert mapped[0] < 1.0

    def test_pairs_raw_sar_with_harmonized_val_not_harmonized_sar(self):
        """Regression test for the bug fixed alongside this test: before
        the fix, fit_sar_to_val_transform built its fitting DataFrame from
        harmonized[[sar_col, val_col]] -- i.e. BOTH columns from the
        harmonize step. That's wrong for this function specifically: ASCAT
        rows' sar_col had already been run through the percent->volumetric
        transform once (down to ISMN's tiny ~0.05-0.6 numeric range), while
        ISMN rows' sar_col stayed raw percent (~7-53). With ASCAT vastly
        outnumbering ISMN -- the realistic case (thousands vs. dozens of
        collocated points) -- the pooled sar_col input was dominated by
        ASCAT's tiny converted sub-population, skewing the percentile
        binning so badly that a genuinely raw field value in ISMN's OWN
        native sar range (~7-53) mapped far outside ISMN's own volumetric
        range (~0.05-0.35).

        After the fix, sar_col always comes straight from collocation_ds
        (raw, untouched by harmonize) and only val_col is harmonized, so
        the fit's x-domain is consistently raw percent regardless of how
        lopsided the group sizes are."""
        from sar_validation.core.statistics import fit_sar_to_val_transform

        rng = np.random.default_rng(7)
        # Realistic imbalance: ISMN in the dozens, ASCAT in the hundreds.
        n_ismn, n_ascat = 18, 300
        ismn_val = rng.uniform(0.05, 0.35, n_ismn)
        ismn_sar = ismn_val * 150.0 + rng.normal(0, 1, n_ismn)
        ascat_val = rng.uniform(10.0, 90.0, n_ascat)
        ascat_sar = ascat_val + rng.normal(0, 1, n_ascat)

        ds = xr.Dataset({
            "sar_sarSSM": xr.DataArray(
                np.concatenate([ismn_sar, ascat_sar]), dims="collocation", attrs={"units": "%"},
            ),
            "val_SOIL_MOISTURE": xr.DataArray(
                np.concatenate([ismn_val, ascat_val]), dims="collocation",
                attrs={"units": "mixed — see val_units"},
            ),
            "val_source": ("collocation", ["ismn"] * n_ismn + ["ascat_ssm"] * n_ascat),
            "val_units": ("collocation", np.array(["1"] * n_ismn + ["%"] * n_ascat)),
        })

        transform = fit_sar_to_val_transform(ds, "sarSSM", "SOIL_MOISTURE")
        assert transform is not None

        # RAW field values spanning ISMN's own raw sar range (~7-53) --
        # exactly what a real, untouched SAR scene pixel looks like --
        # must map into ISMN's own volumetric range (~0.05-0.35), with a
        # little headroom for fit noise. The pre-fix bug mapped the median
        # of this same range to ~0.56 (verified against the pre-fix
        # construction while writing this test) -- well outside ISMN's
        # range -- because ASCAT's tiny converted sar sub-population (300
        # points squeezed into ~0.05-0.6) swamped ISMN's raw ~7-53 points
        # in the pooled percentile binning.
        lo, mid, hi = np.percentile(ismn_sar, [10, 50, 90])
        mapped = transform(np.array([lo, mid, hi]))

        assert np.all(np.isfinite(mapped))
        margin = 0.1
        assert np.all(mapped > ismn_val.min() - margin)
        assert np.all(mapped < ismn_val.max() + margin)

        # The fit must also be monotonic (increasing raw sar -> increasing
        # mapped value) across ISMN's own range -- a sign the x-domain
        # wasn't left bimodal/skewed by mixing raw and converted scales.
        assert mapped[0] < mapped[1] < mapped[2]


class TestRunStatisticsSoilMoistureDispatch:
    def test_soil_moisture_variable_uses_pytesmo_path(self, tmp_path, soil_moisture_collocation_ds):
        from sar_validation.core.statistics import run_statistics

        recipe = Recipe(RecipeConfig(name="test", variable="soil_moisture"))

        with patch.dict("sys.modules", _patched_pytesmo_modules()):
            results = run_statistics(soil_moisture_collocation_ds, recipe, tmp_path)

        assert "sarSSM_vs_SOIL_MOISTURE" in results
        assert "ubrmsd" in results["sarSSM_vs_SOIL_MOISTURE"].data_vars

    def test_wind_variable_does_not_require_pytesmo(self, tmp_path, collocation_ds):
        from sar_validation.core.statistics import run_statistics

        recipe = Recipe(RecipeConfig(name="test", variable="wind"))
        # No pytesmo mock installed — if run_statistics tried to import it
        # for a non-soil-moisture recipe, this would raise ImportError.
        results = run_statistics(collocation_ds, recipe, tmp_path)
        assert "owiWindSpeed_vs_WSPD" in results
        assert "ubrmsd" not in results["owiWindSpeed_vs_WSPD"].data_vars


# ---------------------------------------------------------------------------
# Circular statistics (wind direction)
# ---------------------------------------------------------------------------

@pytest.fixture
def direction_collocation_ds():
    """Direction pairs that straddle the 0°/360° wrap boundary."""
    sar_deg = np.array([10.0, 90.0, 180.0, 270.0, 359.0])
    val_deg = (sar_deg + 2.0) % 360.0  # sar - val should wrap to ~-2° everywhere

    ds = xr.Dataset(
        {
            "sar_owiWindDirection": ("collocation", sar_deg),
            "val_WDIR":             ("collocation", val_deg),
            "val_source":           ("collocation", ["buoy"] * len(sar_deg)),
        }
    )
    return ds


class TestCircularStatistics:
    def test_bias_rmse_correlation_no_warning(self, direction_collocation_ds):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            ds = compute_statistics(direction_collocation_ds, "owiWindDirection", "WDIR")

        # A naive (sar - val) mean would be dominated by the 358° outlier at
        # the wrap boundary; the correct wrapped bias is close to -2°.
        assert abs(float(ds["bias"].values[0]) - (-2.0)) < 1e-6
        assert float(ds["rmse"].values[0]) < 5.0
        assert float(ds["correlation"].values[0]) > 0.9

    def test_constant_direction_group_no_warning(self):
        """A group with zero angular spread must not raise a RuntimeWarning."""
        ds = xr.Dataset(
            {
                "sar_owiWindDirection": ("collocation", [180.0, 180.0]),
                "val_WDIR":             ("collocation", [180.0, 180.0]),
                "val_source":           ("collocation", ["buoy", "buoy"]),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = compute_statistics(ds, "owiWindDirection", "WDIR")
        assert np.isnan(float(result["correlation"].values[0]))


# ---------------------------------------------------------------------------
# run_statistics — platform-type grouping
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_station_collocation_ds():
    """Multiple stations per platform type, plus a scatterometer source."""
    rng = np.random.default_rng(7)

    def _block(n, source, val_ids):
        sar_vals = rng.uniform(2, 14, size=n)
        val_vals = sar_vals + rng.normal(0, 0.5, size=n)
        return sar_vals, val_vals, [source] * n, val_ids

    sar1, val1, src1, id1 = _block(6, "mooring", ["MO_A"] * 3 + ["MO_B"] * 3)
    sar2, val2, src2, id2 = _block(4, "buoy", ["BUOY_X"] * 2 + ["BUOY_Y"] * 2)
    sar3, val3, src3, id3 = _block(50, "scatterometer", ["unknown"] * 50)

    sar_vals = np.concatenate([sar1, sar2, sar3])
    val_vals = np.concatenate([val1, val2, val3])
    sources = src1 + src2 + src3
    val_ids = id1 + id2 + id3

    return xr.Dataset(
        {
            "sar_owiWindSpeed": ("collocation", sar_vals),
            "val_WSPD":         ("collocation", val_vals),
            "val_source":       ("collocation", sources),
            "val_id":           ("collocation", val_ids),
        }
    )


class TestRunStatisticsGrouping:
    def _recipe(self):
        return Recipe(RecipeConfig(name="test", variable="wind"))

    def test_groups_by_platform_type_not_station(self, tmp_path, multi_station_collocation_ds):
        results = run_statistics(multi_station_collocation_ds, self._recipe(), tmp_path)
        stats_ds = results["owiWindSpeed_vs_WSPD"]
        assert set(stats_ds["source"].values) == {"mooring", "buoy", "scatterometer"}

    def test_scatterometer_row_present_with_full_count(self, tmp_path, multi_station_collocation_ds):
        results = run_statistics(multi_station_collocation_ds, self._recipe(), tmp_path)
        stats_ds = results["owiWindSpeed_vs_WSPD"]
        df = stats_ds.to_dataframe()
        assert int(df.loc["scatterometer", "N"]) == 50


# ---------------------------------------------------------------------------
# save_statistics
# ---------------------------------------------------------------------------

class TestSaveStatistics:
    def test_writes_nc_and_csv(self, tmp_path, collocation_ds):
        stats_ds = compute_statistics(collocation_ds, "owiWindSpeed", "WSPD")
        out_path = tmp_path / "stats.nc"
        save_statistics(stats_ds, out_path)
        assert out_path.exists()
        assert (tmp_path / "stats.csv").exists()

    def test_nc_roundtrip(self, tmp_path, collocation_ds):
        stats_ds = compute_statistics(collocation_ds, "owiWindSpeed", "WSPD")
        out_path = tmp_path / "stats.nc"
        save_statistics(stats_ds, out_path)
        loaded = xr.open_dataset(out_path)
        assert "bias" in loaded.data_vars
        assert "rmse" in loaded.data_vars


# ---------------------------------------------------------------------------
# run_statistics_native_units
# ---------------------------------------------------------------------------

class TestRunStatisticsNativeUnits:
    def _make_recipe(self, tmp_path):
        from sar_validation.core.recipe import (
            GeographicBounds,
            Recipe,
            RecipeConfig,
            TemporalBounds,
        )

        cfg = RecipeConfig(
            name="test_native_units",
            variable="soil_moisture",
            geographic_bounds=GeographicBounds(-20.0, 0.0, 35.0, 60.0),
            temporal_bounds=TemporalBounds("2026-01-01", "2026-01-02"),
        )
        return Recipe(config=cfg)

    def test_only_matching_unit_sources_included(self, tmp_path):
        from sar_validation.core.statistics import run_statistics_native_units

        recipe = self._make_recipe(tmp_path)
        collocation_ds = xr.Dataset(
            {
                "sar_sarSSM": (
                    "collocation", np.array([20.0, 30.0, 40.0, 50.0]),
                    {"units": "%"},
                ),
                "val_SOIL_MOISTURE": ("collocation", np.array([25.0, 35.0, 0.15, 0.20])),
                "val_source": ("collocation", np.array(["ascat_ssm", "ascat_ssm", "ismn", "ismn"])),
            },
        )

        results = run_statistics_native_units(collocation_ds, recipe, tmp_path)

        assert "sarSSM_vs_SOIL_MOISTURE" in results
        stats_ds = results["sarSSM_vs_SOIL_MOISTURE"]
        # Only ascat_ssm shares SAR's "%" family — ismn (volumetric) excluded.
        assert list(stats_ds["source"].values) == ["ascat_ssm"]
        assert (tmp_path / "validation_statistics_sarSSM_vs_SOIL_MOISTURE_native_units.nc").exists()
        assert (tmp_path / "validation_statistics_sarSSM_vs_SOIL_MOISTURE_native_units.csv").exists()

    def test_no_matching_sources_produces_no_output(self, tmp_path):
        from sar_validation.core.statistics import run_statistics_native_units

        recipe = self._make_recipe(tmp_path)
        collocation_ds = xr.Dataset(
            {
                "sar_sarSSM": ("collocation", np.array([20.0, 30.0]), {"units": "%"}),
                "val_SOIL_MOISTURE": ("collocation", np.array([0.15, 0.20])),
                "val_source": ("collocation", np.array(["ismn", "ismn"])),
            },
        )

        results = run_statistics_native_units(collocation_ds, recipe, tmp_path)

        assert results == {}
        assert not (tmp_path / "validation_statistics_sarSSM_vs_SOIL_MOISTURE_native_units.nc").exists()

    def test_non_soil_moisture_recipe_returns_empty(self, tmp_path):
        from sar_validation.core.recipe import GeographicBounds, Recipe, RecipeConfig, TemporalBounds
        from sar_validation.core.statistics import run_statistics_native_units

        cfg = RecipeConfig(
            name="test_wind", variable="wind",
            geographic_bounds=GeographicBounds(-20.0, 0.0, 35.0, 60.0),
            temporal_bounds=TemporalBounds("2026-01-01", "2026-01-02"),
        )
        recipe = Recipe(config=cfg)
        collocation_ds = xr.Dataset({"sar_owiWindSpeed": ("collocation", np.array([1.0]))})

        assert run_statistics_native_units(collocation_ds, recipe, tmp_path) == {}


# ---------------------------------------------------------------------------
# _variable_map
# ---------------------------------------------------------------------------

def test_infer_variable_pairs_unknown_variable_raises():
    with pytest.raises(KeyError):
        infer_variable_pairs("invalid_variable")


# ---------------------------------------------------------------------------
# filter_variable_pairs
# ---------------------------------------------------------------------------

def _waves_recipe(swath_mode):
    return Recipe(RecipeConfig(
        name="waves_test",
        variable="waves",
        sar_data=SARDataSpec(swath_mode=swath_mode),
    ))


class TestFilterVariablePairs:
    def test_mixed_mode_uses_oswTotalHs_when_present(self):
        """Regression test: recipe requests [WV, SM] but only WV scenes were
        actually downloaded, so the dataset only has sar_oswTotalHs. This is
        the exact scenario from recipes/waves_example.yaml that produced zero
        statistics before the fix."""
        recipe = _waves_recipe(["WV", "SM"])
        ds = xr.Dataset({
            "sar_oswTotalHs": ("collocation", [1.4, 1.5]),
            "val_VAVH":       ("collocation", [1.42, 1.48]),
            "val_source":     ("collocation", ["altimeter", "altimeter"]),
        })
        pairs = filter_variable_pairs(recipe, ds)
        assert pairs == [("oswTotalHs", "VAVH")]

    def test_falls_back_to_oswHs_when_oswTotalHs_absent(self):
        recipe = _waves_recipe(["WV", "SM"])
        ds = xr.Dataset({
            "sar_oswHs":  ("collocation", [1.4, 1.5]),
            "val_VAVH":   ("collocation", [1.42, 1.48]),
            "val_source": ("collocation", ["altimeter", "altimeter"]),
        })
        pairs = filter_variable_pairs(recipe, ds)
        assert pairs == [("oswHs", "VAVH")]

    def test_owiSignificantWaveHeight_excluded_when_all_nan(self):
        """owiSignificantWaveHeight must NOT be selected when its column is
        entirely NaN — this matches every real product observed so far."""
        recipe = _waves_recipe(["WV", "SM"])
        ds = xr.Dataset({
            "sar_oswTotalHs":               ("collocation", [1.4, 1.5]),
            "sar_owiSignificantWaveHeight": ("collocation", [np.nan, np.nan]),
            "val_VAVH":                     ("collocation", [1.42, 1.48]),
            "val_source":                   ("collocation", ["altimeter", "altimeter"]),
        })
        pairs = filter_variable_pairs(recipe, ds)
        assert pairs == [("oswTotalHs", "VAVH")]

    def test_owiSignificantWaveHeight_additive_when_it_has_data(self):
        """When owiSignificantWaveHeight has at least one real value, stats
        must be produced for BOTH it and the primary variable (oswTotalHs) —
        not just one or the other."""
        recipe = _waves_recipe(["WV", "SM"])
        ds = xr.Dataset({
            "sar_oswTotalHs":               ("collocation", [1.4, 1.5]),
            "sar_owiSignificantWaveHeight": ("collocation", [1.35, np.nan]),
            "val_VAVH":                     ("collocation", [1.42, 1.48]),
            "val_source":                   ("collocation", ["altimeter", "altimeter"]),
        })
        pairs = filter_variable_pairs(recipe, ds)
        assert set(pairs) == {("oswTotalHs", "VAVH"), ("owiSignificantWaveHeight", "VAVH")}

    def test_does_not_double_count_oswTotalHs_and_oswHs(self):
        """oswTotalHs must win outright over oswHs — oswHs must not also
        appear even though its column exists in the dataset."""
        recipe = _waves_recipe(["WV"])
        ds = xr.Dataset({
            "sar_oswTotalHs": ("collocation", [1.4, 1.5]),
            "sar_oswHs":      ("collocation", [1.3, 1.6]),
            "val_VAVH":       ("collocation", [1.42, 1.48]),
            "val_source":     ("collocation", ["altimeter", "altimeter"]),
        })
        pairs = filter_variable_pairs(recipe, ds)
        assert pairs == [("oswTotalHs", "VAVH")]

    def test_multiple_val_vars_cross_single_sar_winner(self):
        """Validation-side candidates are unaffected: every val_var that
        exists still produces its own pair against the one winning sar_var."""
        recipe = _waves_recipe(["WV", "SM"])
        ds = xr.Dataset({
            "sar_oswTotalHs": ("collocation", [1.4, 1.5]),
            "val_VAVH":       ("collocation", [1.42, 1.48]),
            "val_VHM0":       ("collocation", [1.40, 1.50]),
            "val_source":     ("collocation", ["altimeter", "buoy"]),
        })
        pairs = filter_variable_pairs(recipe, ds)
        assert set(pairs) == {("oswTotalHs", "VAVH"), ("oswTotalHs", "VHM0")}

    def test_owiSignificantWaveHeight_alone_when_primary_absent(self):
        """When neither oswTotalHs nor oswHs exists, owiSignificantWaveHeight
        alone is selected as the (only) SAR variable, as long as it has data."""
        recipe = _waves_recipe(["WV", "SM"])
        ds = xr.Dataset({
            "sar_owiSignificantWaveHeight": ("collocation", [1.35, 1.4]),
            "val_VAVH":                     ("collocation", [1.42, 1.48]),
            "val_source":                   ("collocation", ["altimeter", "altimeter"]),
        })
        pairs = filter_variable_pairs(recipe, ds)
        assert pairs == [("owiSignificantWaveHeight", "VAVH")]


class TestFilterVariablePairsSoilMoisture:
    def test_pair_present_when_columns_exist(self):
        recipe = Recipe(RecipeConfig(name="test", variable="soil_moisture"))
        ds = xr.Dataset({
            "sar_sarSSM":         ("collocation", [50.0, 60.0]),
            "val_SOIL_MOISTURE":  ("collocation", [0.2, 0.25]),
        })
        pairs = filter_variable_pairs(recipe, ds)
        assert pairs == [("sarSSM", "SOIL_MOISTURE")]

    def test_pair_absent_when_columns_missing(self):
        recipe = Recipe(RecipeConfig(name="test", variable="soil_moisture"))
        ds = xr.Dataset({"sar_owiWindSpeed": ("collocation", [1.0])})
        pairs = filter_variable_pairs(recipe, ds)
        assert pairs == []


# ---------------------------------------------------------------------------
# Tests for extracted helper functions
# ---------------------------------------------------------------------------


def test_missing_columns_reports_absent_columns():
    ds = xr.Dataset({"sar_x": ("collocation", [1.0])})
    assert _missing_columns(ds, "sar_x", "val_x") == ["val_x"]


def test_missing_columns_empty_when_all_present():
    ds = xr.Dataset({"sar_x": ("collocation", [1.0]), "val_x": ("collocation", [2.0])})
    assert _missing_columns(ds, "sar_x", "val_x") == []


def test_group_by_columns_single_column():
    df = pd.DataFrame({"a": ["x", "x", "y"], "v": [1, 2, 3]})
    groups = _group_by_columns(df, ["a"])
    assert sorted(groups.groups.keys()) == ["x", "y"]


def test_group_by_columns_composite_key():
    df = pd.DataFrame({"a": ["x", "x"], "b": ["1", "2"], "v": [1, 2]})
    groups = _group_by_columns(df, ["a", "b"])
    assert sorted(groups.groups.keys()) == ["x | 1", "x | 2"]


def test_core_metrics_basic():
    sar = np.array([1.0, 2.0, 3.0])
    val = np.array([1.0, 2.0, 4.0])
    record = _core_metrics(sar, val)
    assert record["N"] == 3
    assert record["bias"] == pytest.approx(np.mean(sar - val))
    assert set(record.keys()) == {"N", "bias", "std", "rmse", "correlation", "scatter_index"}


def test_assemble_stats_dataset_shape():
    records = [{"N": 3, "bias": 0.1, "std": 0.2, "rmse": 0.3, "correlation": 0.9, "scatter_index": 0.05}]
    ds = _assemble_stats_dataset(records, ["buoy"], "wind_speed", "WSPD", ["val_source"])
    assert list(ds["source"].values) == ["buoy"]
    assert ds.attrs["group_by"] == "val_source"
    assert float(ds["bias"].values[0]) == pytest.approx(0.1)
