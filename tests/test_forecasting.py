from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from forecasting import MonteCarloForecaster


def _make_df(values):
    """Create a DataFrame matching the format expected by MonteCarloForecaster."""
    dates = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"Completed": values}, index=dates)


class TestRunSimulation:
    def test_raises_when_no_mode_specified(self):
        forecaster = MonteCarloForecaster()
        df = _make_df([1, 2, 3])
        with pytest.raises(ValueError, match="Either num_days or target_count"):
            forecaster.run_simulation(df)

    def test_num_days_returns_four_percentiles(self):
        np.random.seed(42)
        forecaster = MonteCarloForecaster(num_simulations=100)
        df = _make_df([1, 2, 3, 4, 5])
        result = forecaster.run_simulation(df, num_days=10)
        assert result.shape == (4,)

    def test_target_count_returns_four_percentiles(self):
        np.random.seed(42)
        forecaster = MonteCarloForecaster(num_simulations=100)
        df = _make_df([1, 2, 3, 4, 5])
        result = forecaster.run_simulation(df, target_count=20)
        assert result.shape == (4,)

    def test_num_days_percentiles_are_descending(self):
        """For num_days mode, lower percentiles should yield higher counts."""
        np.random.seed(42)
        forecaster = MonteCarloForecaster(num_simulations=1000)
        df = _make_df([1, 2, 3, 4, 5])
        result = forecaster.run_simulation(df, num_days=30)
        # result order: 50th, 25th, 15th, 5th quantiles (descending)
        assert result[0] >= result[1] >= result[2] >= result[3]

    def test_target_count_percentiles_are_ascending(self):
        """For target_count mode, higher percentiles should need more days."""
        np.random.seed(42)
        forecaster = MonteCarloForecaster(num_simulations=1000)
        df = _make_df([1, 2, 3, 4, 5])
        result = forecaster.run_simulation(df, target_count=50)
        # result order: 50th, 75th, 85th, 95th (ascending)
        assert result[0] <= result[1] <= result[2] <= result[3]

    def test_deterministic_with_seed(self):
        df = _make_df([1, 2, 3, 4, 5])
        forecaster = MonteCarloForecaster(num_simulations=500)

        np.random.seed(123)
        result1 = forecaster.run_simulation(df, num_days=14)

        np.random.seed(123)
        result2 = forecaster.run_simulation(df, num_days=14)

        np.testing.assert_array_equal(result1, result2)

    def test_all_zeros_with_target_count_raises(self):
        """If no work is ever done, raise a clear error."""
        np.random.seed(42)
        forecaster = MonteCarloForecaster(num_simulations=100)
        df = _make_df([0, 0, 0, 0])
        with pytest.raises(ValueError, match="zero historical throughput"):
            forecaster.run_simulation(df, target_count=10)


class TestForecastBacklog:
    def test_returns_days_and_dates(self):
        np.random.seed(42)
        forecaster = MonteCarloForecaster(num_simulations=100)
        df = _make_df([1, 2, 3, 4, 5])
        days, dates = forecaster.forecast_backlog(df, backlog_size=20)
        assert len(days) == 4
        assert len(dates) == 4

    def test_dates_are_date_objects(self):
        np.random.seed(42)
        forecaster = MonteCarloForecaster(num_simulations=100)
        df = _make_df([1, 2, 3, 4, 5])
        _, dates = forecaster.forecast_backlog(df, backlog_size=20)
        for d in dates:
            assert hasattr(d, "year")  # date objects have .year

    def test_dates_are_in_future(self):
        np.random.seed(42)
        forecaster = MonteCarloForecaster(num_simulations=100)
        df = _make_df([1, 2, 3, 4, 5])
        _, dates = forecaster.forecast_backlog(df, backlog_size=20)
        today = datetime.now().date()
        for d in dates:
            assert d >= today
