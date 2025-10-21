"""
Basic Usage Examples for Monte Carlo Forecasting

This file demonstrates common forecasting scenarios using the Monte Carlo skill.
"""

import sys
import os

# Add parent directory to path to import forecaster
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from forecaster import (
    MonteCarloForecaster,
    prepare_dataframe_from_daily_data,
    prepare_dataframe_from_csv
)


def example_1_csv_file():
    """Example: Load data from CSV and forecast completion time"""
    print("\n" + "="*60)
    print("Example 1: Forecast completion time from CSV data")
    print("="*60)

    # Load data from CSV file
    csv_path = os.path.join(os.path.dirname(__file__), 'example_csv_data.csv')
    df = prepare_dataframe_from_csv(csv_path, date_column='date', completed_column='completed')

    print(f"\nLoaded {len(df)} days of historical data")
    print(f"Average daily completion: {df['Completed'].mean():.1f} items")

    # Forecast: When will we complete 50 items?
    forecaster = MonteCarloForecaster(num_simulations=10000)
    days = forecaster.forecast_completion_time(df, item_count=50)

    print(f"\nForecast for completing 50 items:")
    print(f"  50% confidence: {days[0]:.0f} days")
    print(f"  75% confidence: {days[1]:.0f} days")
    print(f"  85% confidence: {days[2]:.0f} days")
    print(f"  95% confidence: {days[3]:.0f} days")


def example_2_manual_data():
    """Example: Create forecast from manually entered data"""
    print("\n" + "="*60)
    print("Example 2: Forecast from manual data entry")
    print("="*60)

    # Manual data entry
    dates = [
        '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
        '2024-01-06', '2024-01-07', '2024-01-08', '2024-01-09', '2024-01-10',
        '2024-01-11', '2024-01-12', '2024-01-13', '2024-01-14', '2024-01-15',
        '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19', '2024-01-20'
    ]
    completed = [5, 3, 7, 2, 6, 4, 0, 5, 8, 3, 6, 4, 7, 0, 5, 6, 4, 8, 3, 7]

    df = prepare_dataframe_from_daily_data(dates, completed)

    # Forecast: How many items in 30 days?
    forecaster = MonteCarloForecaster()
    items = forecaster.forecast_completed_items(df, num_days=30)

    print(f"\nForecast for items completed in 30 days:")
    print(f"  50% confidence: {items[0]:.0f} items (median)")
    print(f"  75% confidence: {items[1]:.0f} items (conservative)")
    print(f"  85% confidence: {items[2]:.0f} items (more conservative)")
    print(f"  95% confidence: {items[3]:.0f} items (very conservative)")


def example_3_backlog_forecast():
    """Example: Forecast backlog completion with dates"""
    print("\n" + "="*60)
    print("Example 3: Backlog completion forecast with dates")
    print("="*60)

    # Load data
    csv_path = os.path.join(os.path.dirname(__file__), 'example_csv_data.csv')
    df = prepare_dataframe_from_csv(csv_path, date_column='date', completed_column='completed')

    # Forecast backlog
    forecaster = MonteCarloForecaster()
    backlog_size = 100
    days, dates = forecaster.forecast_backlog(df, backlog_size=backlog_size)

    print(f"\nForecast for completing {backlog_size} items backlog:")
    print(f"  50% confidence: {days[0]:.0f} days ({dates[0]})")
    print(f"  75% confidence: {days[1]:.0f} days ({dates[1]})")
    print(f"  85% confidence: {days[2]:.0f} days ({dates[2]})")
    print(f"  95% confidence: {days[3]:.0f} days ({dates[3]})")


def example_4_multiple_scenarios():
    """Example: Compare multiple scenarios"""
    print("\n" + "="*60)
    print("Example 4: Compare multiple backlog scenarios")
    print("="*60)

    # Load data
    csv_path = os.path.join(os.path.dirname(__file__), 'example_csv_data.csv')
    df = prepare_dataframe_from_csv(csv_path, date_column='date', completed_column='completed')

    forecaster = MonteCarloForecaster()

    print("\nComparison at 75% confidence level:")
    print(f"{'Backlog Size':<15} {'Days to Complete':<20} {'Completion Date'}")
    print("-" * 60)

    for backlog_size in [25, 50, 75, 100, 150]:
        days, dates = forecaster.forecast_backlog(df, backlog_size=backlog_size)
        # Use 75th percentile (index 1)
        print(f"{backlog_size:<15} {days[1]:<20.0f} {dates[1]}")


def example_5_custom_simulations():
    """Example: Using custom simulation count for higher accuracy"""
    print("\n" + "="*60)
    print("Example 5: Custom simulation count")
    print("="*60)

    csv_path = os.path.join(os.path.dirname(__file__), 'example_csv_data.csv')
    df = prepare_dataframe_from_csv(csv_path, date_column='date', completed_column='completed')

    print("\nComparing different simulation counts:")

    for sim_count in [1000, 10000, 50000]:
        forecaster = MonteCarloForecaster(num_simulations=sim_count)
        days = forecaster.forecast_completion_time(df, item_count=50)
        print(f"\n{sim_count:,} simulations:")
        print(f"  Median: {days[0]:.1f} days")
        print(f"  95th percentile: {days[3]:.1f} days")


if __name__ == '__main__':
    print("\nMonte Carlo Forecasting - Usage Examples")
    print("=========================================")

    example_1_csv_file()
    example_2_manual_data()
    example_3_backlog_forecast()
    example_4_multiple_scenarios()
    example_5_custom_simulations()

    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60 + "\n")
