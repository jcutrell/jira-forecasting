import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_cumulative_flow(df):
    plt.figure(figsize=(12, 6))
    plt.plot(
        df.index, df["Cumulative Completed Items"], label="Cumulative Completed Items"
    )
    plt.title("Cumulative Flow Diagram")
    plt.xlabel("Date")
    plt.ylabel("Number of Items")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_monte_carlo_results(forecasted_values, title, xlabel):
    plt.figure(figsize=(10, 6))
    percentiles = [50, 75, 85, 95]
    plt.bar(percentiles, forecasted_values, color="skyblue")
    plt.title(title)
    plt.xlabel("Percentile")
    plt.ylabel(xlabel)
    plt.xticks(percentiles)
    for i, v in enumerate(forecasted_values):
        plt.text(percentiles[i], v, f"{v:.0f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.show()


def plot_cycle_time_distribution(cycle_times):
    plt.figure(figsize=(10, 6))
    plt.hist([ct.days for ct in cycle_times], bins=20, edgecolor="black")
    plt.title("Cycle Time Distribution")
    plt.xlabel("Cycle Time (days)")
    plt.ylabel("Frequency")
    plt.axvline(
        np.median([ct.days for ct in cycle_times]),
        color="r",
        linestyle="dashed",
        linewidth=2,
        label="Median",
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_completion_forecast(forecast_dates, actual_dates):
    plt.figure(figsize=(12, 6))
    plt.plot(actual_dates, forecast_dates, marker="o")
    plt.title("Completion Date Forecast Over Time")
    plt.xlabel("Forecast Date")
    plt.ylabel("Projected Completion Date")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_throughput_trend(df):
    weekly_throughput = df["Completed Items"].resample("W").sum()
    plt.figure(figsize=(12, 6))
    plt.plot(weekly_throughput.index, weekly_throughput.values, marker="o")
    plt.title("Weekly Throughput Trend")
    plt.xlabel("Date")
    plt.ylabel("Completed Items per Week")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
