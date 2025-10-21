---
name: Monte Carlo Forecasting
description: Generate probabilistic forecasts using Monte Carlo simulation on historical completion data. Works with any time-series data including tickets, story points, or custom metrics.
---

# Monte Carlo Forecasting Skill

## Overview

This skill provides Monte Carlo simulation-based forecasting capabilities for any time-series completion data. It generates probabilistic forecasts by sampling from historical daily completion rates and provides confidence intervals at multiple percentiles (50%, 75%, 85%, 95%).

## When to Use This Skill

Use this skill when the user needs to:
- Forecast completion dates for projects or backlogs
- Predict how much work can be completed in a given timeframe
- Generate probabilistic estimates with confidence intervals
- Analyze throughput trends and make data-driven predictions
- Work with any completion metric (tickets, story points, tasks, etc.)

## Data Requirements

The forecasting algorithm requires historical completion data in the following format:

**Required DataFrame Structure:**
```python
# DatetimeIndex with daily dates
# 'Completed' column with numeric values (items/points completed per day)

Example:
Date        | Completed
------------|----------
2024-01-01  | 5
2024-01-02  | 3
2024-01-03  | 7
2024-01-04  | 2
```

## Core Capabilities

### 1. Forecast Completion Time
**Question:** "When will we finish N items?"

```python
from forecaster import MonteCarloForecaster

forecaster = MonteCarloForecaster(num_simulations=10000)
days = forecaster.forecast_completion_time(df, item_count=50)

# Returns [50th, 75th, 85th, 95th percentile] days
# Example: [12, 15, 17, 21]
# Interpretation: 50% chance in 12 days, 95% chance in 21 days
```

### 2. Forecast Completed Items
**Question:** "How many items will we complete in N days?"

```python
items = forecaster.forecast_completed_items(df, num_days=30)

# Returns [50th, 75th, 85th, 95th percentile] item counts
# Example: [45, 38, 35, 28]
# Interpretation: 50% chance to complete 45+ items in 30 days
```

### 3. Forecast Backlog Completion
**Question:** "When will we finish our entire backlog?"

```python
days, dates = forecaster.forecast_backlog(df, backlog_size=100)

# Returns:
# - days: [50th, 75th, 85th, 95th percentile] days to completion
# - dates: Corresponding calendar dates
```

## Instructions for Claude

When a user requests forecasting or predictions:

1. **Identify the Data Source**
   - Determine what data is available (CSV, database, API, etc.)
   - Confirm the data contains completion information over time

2. **Prepare the Data**
   - Create or load a DataFrame with daily completion counts
   - Ensure it has a 'Completed' column with numeric values
   - Use helper functions if needed:
     - `prepare_dataframe_from_daily_data()` for lists
     - `prepare_dataframe_from_csv()` for CSV files

3. **Clarify the Forecast Type**
   - **Time-based:** "How many X in Y days?" → use `forecast_completed_items()`
   - **Count-based:** "When will we finish X items?" → use `forecast_completion_time()`
   - **Backlog:** "When will backlog be done?" → use `forecast_backlog()`

4. **Run the Simulation**
   - Create a `MonteCarloForecaster` instance
   - Call the appropriate method
   - Return results with confidence intervals

5. **Interpret Results**
   - Explain what each percentile means
   - Provide actionable insights
   - Suggest visualizations if helpful

## Data Source Examples

### Example 1: CSV File
```python
from forecaster import prepare_dataframe_from_csv, MonteCarloForecaster

# Load data from CSV
df = prepare_dataframe_from_csv(
    'completion_data.csv',
    date_column='date',
    completed_column='items_done'
)

# Run forecast
forecaster = MonteCarloForecaster()
days = forecaster.forecast_completion_time(df, item_count=50)
print(f"50% confidence: {days[0]} days")
print(f"95% confidence: {days[3]} days")
```

### Example 2: Manual Data Entry
```python
from forecaster import prepare_dataframe_from_daily_data, MonteCarloForecaster

# Create DataFrame from lists
dates = ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
completed = [5, 3, 7, 4, 6]

df = prepare_dataframe_from_daily_data(dates, completed)

# Forecast
forecaster = MonteCarloForecaster()
items = forecaster.forecast_completed_items(df, num_days=10)
print(f"Expected items in 10 days: {items[0]} (50% confidence)")
```

### Example 3: Jira Data (Using Existing Integration)
```python
# If working with Jira data, leverage existing jira_data.py module
from jira_data import JiraDataManager

jira_manager = JiraDataManager(config)
df_items, df_points = jira_manager.prepare_data(tickets)

# Now use the generic forecaster
forecaster = MonteCarloForecaster()
days = forecaster.forecast_completion_time(df_items, item_count=30)
```

### Example 4: Database Query
```python
import pandas as pd
from forecaster import MonteCarloForecaster

# Query database
query = """
    SELECT DATE(completed_at) as date, COUNT(*) as completed
    FROM tasks
    WHERE status = 'done'
    GROUP BY DATE(completed_at)
    ORDER BY date
"""
df = pd.read_sql(query, connection)
df.set_index('date', inplace=True)
df.columns = ['Completed']

# Forecast
forecaster = MonteCarloForecaster()
result = forecaster.forecast_backlog(df, backlog_size=200)
```

## Understanding the Results

### Percentile Interpretation

- **50th percentile:** Median outcome, 50% probability of achieving this or better
- **75th percentile:** Conservative estimate, 75% probability of success
- **85th percentile:** More conservative, 85% probability of success
- **95th percentile:** High confidence estimate, 95% probability of success

### When to Use Which Percentile

- **Planning commitments:** Use 75th-85th percentile for reliable estimates
- **Internal goals:** Use 50th percentile for stretch targets
- **External commitments:** Use 85th-95th percentile to avoid missed deadlines
- **Risk assessment:** Show full range to communicate uncertainty

## Advanced Usage

### Custom Simulation Count
```python
# More simulations = more accuracy but slower
forecaster = MonteCarloForecaster(num_simulations=50000)
```

### Multiple Scenarios
```python
# Compare different backlog sizes
for size in [50, 100, 150]:
    days, dates = forecaster.forecast_backlog(df, backlog_size=size)
    print(f"{size} items: {days[1]} days (75% confidence)")
```

### Handling Missing Data
```python
# Fill missing dates with zero completions
df = df.asfreq('D', fill_value=0)
```

## Limitations and Considerations

1. **Historical Data Quality**
   - Requires sufficient historical data (recommended: 30+ data points)
   - Assumes future work patterns similar to past patterns
   - Outliers in historical data will affect simulations

2. **Assumptions**
   - Daily completion rates are independent
   - Work capacity remains relatively stable
   - No major process changes expected

3. **When NOT to Use**
   - Very small datasets (< 10 data points)
   - Highly seasonal work patterns without adjustment
   - Significant known changes in team capacity

## Troubleshooting

**Error: "DataFrame must contain a 'Completed' column"**
- Ensure your DataFrame has a column named exactly 'Completed'
- Use helper functions to rename columns if needed

**Error: "Average daily completion rate is zero"**
- Check that historical data contains non-zero values
- Verify data is properly loaded and formatted

**Unrealistic forecasts**
- Verify historical data quality
- Check for data entry errors or anomalies
- Consider filtering outliers

## Integration Tips

- The forecaster is completely data-source agnostic
- Works with any system that can produce daily completion counts
- Can be integrated with existing data pipelines
- Results can be exported for visualization or reporting

## References

Monte Carlo simulation uses random sampling from historical data to model uncertainty and generate probabilistic forecasts. The method is particularly effective for forecasting when:
- Historical data shows variability
- Precise probability estimates are needed
- Risk assessment is important
- Multiple scenarios need to be compared
