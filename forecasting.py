import numpy as np
import pandas as pd
from typing import List, Tuple
from datetime import datetime, timedelta


class MonteCarloForecaster:
    def __init__(self, num_simulations: int = 10000):
        self.num_simulations = num_simulations

    def run_simulation(
        self, df: pd.DataFrame, num_days: int = None, target_count: float = None
    ) -> np.ndarray:
        completed_per_day = df["Completed"].values

        if num_days is None and target_count is None:
            raise ValueError("Either num_days or target_count must be specified")

        if num_days is None:
            num_days = int(2 * target_count / np.mean(completed_per_day))

        simulations = np.random.choice(
            completed_per_day, size=(self.num_simulations, num_days)
        )
        simulations = np.cumsum(simulations, axis=1)

        if target_count is not None:
            days_to_target = np.argmax(simulations >= target_count, axis=1)
            return np.quantile(days_to_target, [0.5, 0.75, 0.85, 0.95])
        else:
            return np.quantile(
                simulations, [0.5, 0.25, 0.15, 0.05], axis=0, method="lower"
            )[:, -1]

    def forecast_completed_items(self, df: pd.DataFrame, num_days: int) -> List[float]:
        return self.run_simulation(df, num_days=num_days)

    def forecast_completion_time(
        self, df: pd.DataFrame, item_count: float
    ) -> List[float]:
        return self.run_simulation(df, target_count=item_count)

    def forecast_backlog(
        self, df: pd.DataFrame, backlog_size: float
    ) -> Tuple[List[float], List[datetime]]:
        days_to_target = self.run_simulation(df, target_count=backlog_size)
        completion_dates = [
            (datetime.now() + timedelta(days=int(days))).date()
            for days in days_to_target
        ]
        return days_to_target, completion_dates
