# Monte Carlo Forecasting Skill

A generic, data-source-agnostic Monte Carlo forecasting skill for Claude Code.

## Overview

This skill enables probabilistic forecasting using Monte Carlo simulation on any time-series completion data. It's completely generic and works with:

- Jira tickets
- GitHub issues
- Story points or effort metrics
- Custom task tracking systems
- Any numeric completion metric tracked over time

## Installation

This skill is ready to use once placed in the `.claude/skills/montecarlo-forecasting/` directory.

### Dependencies

The skill requires:
- Python 3.7+
- numpy
- pandas

Install dependencies:
```bash
pip install numpy pandas
```

## Quick Start

### 1. Prepare Your Data

Your data needs to be in a DataFrame with a 'Completed' column:

```python
from forecaster import prepare_dataframe_from_csv

df = prepare_dataframe_from_csv('your_data.csv',
                                 date_column='date',
                                 completed_column='items')
```

### 2. Run a Forecast

```python
from forecaster import MonteCarloForecaster

forecaster = MonteCarloForecaster()

# When will we finish 50 items?
days = forecaster.forecast_completion_time(df, item_count=50)
print(f"75% confidence: {days[1]} days")
```

## Examples

See the `examples/` directory for complete working examples:

- `basic_usage.py` - Comprehensive examples covering all use cases
- `example_csv_data.csv` - Sample data file

Run the examples:
```bash
cd examples/
python basic_usage.py
```

## How It Works

The Monte Carlo forecaster:

1. Takes historical daily completion data
2. Randomly samples from historical rates (with replacement)
3. Runs thousands of simulations of future work
4. Calculates percentiles for confidence intervals

### Why Monte Carlo?

- **Handles variability:** Accounts for natural fluctuations in work completion
- **Probabilistic:** Provides confidence intervals, not just single point estimates
- **Data-driven:** Based on your actual historical performance
- **No assumptions:** Doesn't assume normal distributions or linear trends

## File Structure

```
montecarlo-forecasting/
├── Skill.md              # Main skill definition (required)
├── README.md             # This file
├── forecaster.py         # Core forecasting module
└── examples/
    ├── basic_usage.py    # Complete usage examples
    └── example_csv_data.csv  # Sample data
```

## Data Format

The forecaster expects a pandas DataFrame with:

- **Index:** DatetimeIndex (daily frequency recommended)
- **Column:** 'Completed' - numeric values representing daily completion counts

Example:
```
Date        | Completed
------------|----------
2024-01-01  | 5
2024-01-02  | 3
2024-01-03  | 7
```

## API Reference

### MonteCarloForecaster

Main forecaster class.

**Methods:**

- `forecast_completion_time(df, item_count)` - Forecast days to complete N items
- `forecast_completed_items(df, num_days)` - Forecast items completed in N days
- `forecast_backlog(df, backlog_size)` - Forecast backlog completion with dates

### Helper Functions

- `prepare_dataframe_from_csv(filepath, date_column, completed_column)` - Load from CSV
- `prepare_dataframe_from_daily_data(dates, completed_counts)` - Create from lists

## Integration with Existing Systems

### Jira

If using the existing Jira integration in this repository:

```python
from jira_data import JiraDataManager
from forecaster import MonteCarloForecaster

# Get Jira data
jira_manager = JiraDataManager(config)
df_items, df_points = jira_manager.prepare_data(tickets)

# Forecast with generic forecaster
forecaster = MonteCarloForecaster()
days = forecaster.forecast_completion_time(df_items, item_count=30)
```

### Custom Data Sources

For any custom source, just create a DataFrame with the required format:

```python
import pandas as pd

# Your custom data loading logic
df = your_data_source.get_daily_completions()

# Rename columns to match expected format
df.rename(columns={'your_date_col': 'Date',
                   'your_count_col': 'Completed'}, inplace=True)
df.set_index('Date', inplace=True)

# Now use the forecaster
forecaster = MonteCarloForecaster()
result = forecaster.forecast_completion_time(df, item_count=50)
```

## Best Practices

1. **Sufficient Data:** Use at least 30 days of historical data for reliable forecasts
2. **Consistent Patterns:** Works best when future work patterns resemble the past
3. **Regular Updates:** Rerun forecasts regularly as new data becomes available
4. **Multiple Percentiles:** Show range of outcomes to communicate uncertainty
5. **Validate Assumptions:** Check that historical data is representative

## Limitations

- Assumes future work patterns similar to historical patterns
- Doesn't account for known future changes (holidays, team changes, etc.)
- Requires sufficient historical data (minimum 10-15 data points)
- Daily completion rates assumed to be independent

## Customization

### Adjust Simulation Count

More simulations = more accuracy but slower:

```python
forecaster = MonteCarloForecaster(num_simulations=50000)
```

### Custom Percentiles

Modify the `run_simulation()` method to return different percentiles.

## Troubleshooting

**No 'Completed' column error:**
- Ensure your DataFrame has a column named exactly 'Completed'
- Use `df.rename(columns={'old_name': 'Completed'})` if needed

**Unrealistic forecasts:**
- Check historical data for errors or anomalies
- Ensure data covers a representative time period
- Consider filtering outliers

**Low accuracy:**
- Increase number of simulations
- Ensure sufficient historical data
- Verify data quality

## Contributing

To extend this skill:

1. Add new forecasting methods to `forecaster.py`
2. Update `Skill.md` with new instructions
3. Add examples demonstrating new capabilities

## License

This skill is part of the jira-forecasting repository.

## Further Reading

- [Monte Carlo Simulation](https://en.wikipedia.org/wiki/Monte_Carlo_method)
- [Probabilistic Forecasting](https://www.agilealliance.org/resources/experience-reports/monte-carlo-simulation-for-agile-projects/)
- [When Will It Be Done?](https://leanpub.com/whenwillitbedone) - Book on probabilistic forecasting
