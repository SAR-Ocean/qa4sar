"""Tests for sar_validation.core.statistics."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from sar_validation.core._variable_map import filter_variable_pairs, infer_variable_pairs
from sar_validation.core.recipe import Recipe, RecipeConfig, SARDataSpec
from sar_validation.core.statistics import compute_statistics, run_statistics, save_statistics

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
    def test_returns_dataset(self, collocation_ds):
        ds = compute_statistics(collocation_ds, "owiWindSpeed", "WSPD")
        assert isinstance(ds, xr.Dataset)

    def test_expected_metrics(self, collocation_ds):
        ds = compute_statistics(collocation_ds, "owiWindSpeed", "WSPD")
        for metric in ("N", "bias", "std", "rmse", "correlation", "scatter_index"):
            assert metric in ds.data_vars, f"Missing metric: {metric}"

    def test_sources_dimension(self, collocation_ds):
        ds = compute_statistics(collocation_ds, "owiWindSpeed", "WSPD")
        assert "source" in ds.dims
        assert set(ds["source"].values) == {"mooring", "buoy"}

    def test_bias_near_zero(self, collocation_ds):
        ds = compute_statistics(collocation_ds, "owiWindSpeed", "WSPD")
        # Synthetic data has small noise so bias should be small
        assert abs(float(ds["bias"].mean())) < 1.0

    def test_rmse_positive(self, collocation_ds):
        ds = compute_statistics(collocation_ds, "owiWindSpeed", "WSPD")
        assert (ds["rmse"].values >= 0).all()

    def test_correlation_in_range(self, collocation_ds):
        ds = compute_statistics(collocation_ds, "owiWindSpeed", "WSPD")
        corr = ds["correlation"].values
        assert np.all((corr >= -1) & (corr <= 1))

    def test_missing_var_returns_none(self, collocation_ds):
        result = compute_statistics(collocation_ds, "owiWaveHeight", "VHM0")
        assert result is None

    def test_n_correct(self, collocation_ds):
        ds = compute_statistics(collocation_ds, "owiWindSpeed", "WSPD")
        total_n = int(ds["N"].sum())
        assert total_n == 40


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


def _fake_ubrmsd(sar_vals, val_vals):
    diff = sar_vals - sar_vals.mean() - (val_vals - val_vals.mean())
    return float(np.sqrt(np.mean(diff ** 2)))


def _patched_pytesmo_modules():
    fake_scaling = MagicMock()
    fake_scaling.scale = _FakeCdfMatchScale()
    fake_metrics = MagicMock()
    fake_metrics.ubrmsd = _fake_ubrmsd
    fake_pytesmo = MagicMock()
    return {
        "pytesmo": fake_pytesmo,
        "pytesmo.scaling": fake_scaling,
        "pytesmo.metrics": fake_metrics,
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
    def test_bias_uses_wrapped_difference(self, direction_collocation_ds):
        ds = compute_statistics(direction_collocation_ds, "owiWindDirection", "WDIR")
        # A naive (sar - val) mean would be dominated by the 358° outlier at
        # the wrap boundary; the correct wrapped bias is close to -2°.
        assert abs(float(ds["bias"].values[0]) - (-2.0)) < 1e-6

    def test_rmse_small_despite_wrap(self, direction_collocation_ds):
        ds = compute_statistics(direction_collocation_ds, "owiWindDirection", "WDIR")
        assert float(ds["rmse"].values[0]) < 5.0

    def test_correlation_near_one_for_rotated_series(self, direction_collocation_ds):
        ds = compute_statistics(direction_collocation_ds, "owiWindDirection", "WDIR")
        assert float(ds["correlation"].values[0]) > 0.9

    def test_no_runtime_warning(self, direction_collocation_ds):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            compute_statistics(direction_collocation_ds, "owiWindDirection", "WDIR")

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
# _variable_map
# ---------------------------------------------------------------------------

class TestVariableMap:
    def test_wind_pairs(self):
        pairs = infer_variable_pairs("wind")
        assert ("owiWindSpeed", "WSPD") in pairs
        assert ("owiWindDirection", "WDIR") in pairs

    def test_currents_pairs(self):
        pairs = infer_variable_pairs("currents")
        assert ("rvlRadVel", "rvlRadVel_projection") in pairs

    def test_waves_pairs(self):
        pairs = infer_variable_pairs("waves")
        # WV wave validation compares the integrated total significant wave
        # height (oswTotalHs); the partitioned oswHs remains as a legacy pair.
        assert ("oswTotalHs", "VHM0") in pairs
        assert ("oswHs", "VHM0") in pairs

    def test_soil_moisture_pairs(self):
        pairs = infer_variable_pairs("soil_moisture")
        assert pairs == [("sarSSM", "SOIL_MOISTURE")]

    def test_unknown_raises(self):
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
