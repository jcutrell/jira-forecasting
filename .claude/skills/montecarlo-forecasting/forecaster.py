"""
Generic Monte Carlo Forecasting Module

This module provides Monte Carlo simulation-based forecasting capabilities
for any time-series data with daily completion metrics.

The forecaster is data-source agnostic and works with:
- Issue/ticket completion rates
- Story points or effort metrics
- Any numeric completion metric tracked over time
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Union, Optional
from datetime import datetime, timedelta


class MonteCarloForecaster:
    """
    A generic Monte Carlo forecaster that uses historical completion data
    to generate probabilistic forecasts.

    The forecaster samples from historical daily completion rates to simulate
    future outcomes and provides confidence intervals at multiple percentiles.
    """

    def __init__(self, num_simulations: int = 10000):
        """
        Initialize the forecaster.

        Args:
            num_simulations: Number of Monte Carlo simulations to run (default: 10000)
        """
        self.num_simulations = num_simulations

    def run_simulation(
        self,
        df: pd.DataFrame,
        num_days: Optional[int] = None,
        target_count: Optional[float] = None
    ) -> np.ndarray:
        """
        Run Monte Carlo simulation using historical completion data.

        Args:
            df: DataFrame with a 'Completed' column containing daily completion counts
            num_days: Number of days to forecast (required if target_count is None)
            target_count: Target number of items to complete (required if num_days is None)

        Returns:
            Array of percentiles [50th, 75th, 85th, 95th] representing confidence levels

        Raises:
            ValueError: If neither num_days nor target_count is specified
            ValueError: If DataFrame doesn't contain 'Completed' column
        """
        if 'Completed' not in df.columns:
            raise ValueError("DataFrame must contain a 'Completed' column")

        completed_per_day = df['Completed'].values

        if num_days is None and target_count is None:
            raise ValueError("Either num_days or target_count must be specified")

        # Estimate simulation horizon if not provided
        if num_days is None:
            avg_daily_rate = np.mean(completed_per_day)
            if avg_daily_rate == 0:
                raise ValueError("Average daily completion rate is zero - cannot forecast")
            num_days = int(2 * target_count / avg_daily_rate)

        # Run simulations: randomly sample from historical rates
        simulations = np.random.choice(
            completed_per_day,
            size=(self.num_simulations, num_days)
        )
        simulations = np.cumsum(simulations, axis=1)

        # Calculate percentiles based on forecast type
        if target_count is not None:
            # Forecast: "When will we complete N items?"
            days_to_target = np.argmax(simulations >= target_count, axis=1)
            return np.quantile(days_to_target, [0.5, 0.75, 0.85, 0.95])
        else:
            # Forecast: "How many items will we complete in N days?"
            return np.quantile(
                simulations, [0.5, 0.25, 0.15, 0.05], axis=0, method='lower'
            )[:, -1]

    def forecast_completed_items(
        self,
        df: pd.DataFrame,
        num_days: int
    ) -> List[float]:
        """
        Forecast how many items will be completed in a given timeframe.

        Args:
            df: DataFrame with 'Completed' column
            num_days: Number of days to forecast

        Returns:
            List of [50th, 75th, 85th, 95th percentile] completion counts
        """
        return self.run_simulation(df, num_days=num_days)

    def forecast_completion_time(
        self,
        df: pd.DataFrame,
        item_count: float
    ) -> List[float]:
        """
        Forecast how many days it will take to complete a target number of items.

        Args:
            df: DataFrame with 'Completed' column
            item_count: Target number of items to complete

        Returns:
            List of [50th, 75th, 85th, 95th percentile] days to completion
        """
        return self.run_simulation(df, target_count=item_count)

    def forecast_backlog(
        self,
        df: pd.DataFrame,
        backlog_size: float,
        start_date: Optional[datetime] = None
    ) -> Tuple[List[float], List[datetime]]:
        """
        Forecast when a backlog of items will be completed.

        Args:
            df: DataFrame with 'Completed' column
            backlog_size: Number of items in the backlog
            start_date: Starting date for forecast (default: today)

        Returns:
            Tuple of (days_to_completion, completion_dates) at each percentile
        """
        if start_date is None:
            start_date = datetime.now()

        days_to_target = self.run_simulation(df, target_count=backlog_size)
        completion_dates = [
            (start_date + timedelta(days=int(days))).date()
            for days in days_to_target
        ]
        return days_to_target, completion_dates


def prepare_dataframe_from_daily_data(
    dates: List[Union[str, datetime]],
    completed_counts: List[float]
) -> pd.DataFrame:
    """
    Helper function to create a properly formatted DataFrame from daily data.

    Args:
        dates: List of dates (as strings or datetime objects)
        completed_counts: List of completion counts corresponding to each date

    Returns:
        DataFrame with DatetimeIndex and 'Completed' column

    Example:
        >>> dates = ['2024-01-01', '2024-01-02', '2024-01-03']
        >>> counts = [5, 3, 7]
        >>> df = prepare_dataframe_from_daily_data(dates, counts)
    """
    df = pd.DataFrame({
        'Date': pd.to_datetime(dates),
        'Completed': completed_counts
    })
    df.set_index('Date', inplace=True)
    return df


def prepare_dataframe_from_csv(
    filepath: str,
    date_column: str = 'Date',
    completed_column: str = 'Completed'
) -> pd.DataFrame:
    """
    Helper function to load and prepare DataFrame from CSV file.

    Args:
        filepath: Path to CSV file
        date_column: Name of the date column
        completed_column: Name of the completed items column

    Returns:
        DataFrame with DatetimeIndex and 'Completed' column

    Example:
        >>> df = prepare_dataframe_from_csv('data.csv', 'date', 'items_completed')
    """
    df = pd.read_csv(filepath)
    df[date_column] = pd.to_datetime(df[date_column])
    df.set_index(date_column, inplace=True)
    df.rename(columns={completed_column: 'Completed'}, inplace=True)
    return df[['Completed']]
