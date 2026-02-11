# Jira Forecasting

Monte Carlo simulation tool for forecasting Jira project completion dates, throughput, and cycle times.

## What it does

- Connects to Jira Cloud and pulls historical ticket data from a saved filter
- Calculates cycle time statistics per workflow status
- Runs Monte Carlo simulations (10,000 iterations) to produce probabilistic forecasts
- Supports forecasting by item count or story points
- Provides per-contributor metrics and backlog completion date estimates

## Setup

### Prerequisites

- Python 3.9+
- A Jira Cloud instance with API access

### Install dependencies

```
pip install -r requirements.txt
```

### Configure Jira credentials

Copy the example config and fill in your details:

```
cp config.ini.example config.ini
```

Edit `config.ini`:

```ini
[Jira]
email = your-email@example.com
api_key = your-jira-api-token
base_url = https://your-domain.atlassian.net
```

You can generate an API token at https://id.atlassian.com/manage-profile/security/api-tokens.

## Usage

```
python main.py
```

The tool will prompt you to:

1. Specify which Jira status represents "Done"
2. Select a Jira filter for historical data
3. Optionally remove statistical outliers (IQR method)
4. Optionally combine cycle times for specific statuses

Then presents an interactive menu:

- **Forecast completed work** -- project how many items/points will be completed over various time periods (14-136 days)
- **Forecast completion time** -- estimate days needed to complete a target amount of work
- **Forecast backlog completion** -- predict completion dates for an unresolved backlog with confidence intervals
- **Individual contributor metrics** -- view per-person cycle times and throughput

## Running tests

```
pip install pytest
python -m pytest tests/ -v
```

## Project structure

```
main.py                  # CLI entry point and menu system
jira_data.py             # Jira API client and data processing
jira_statistics.py       # Statistical analysis and reporting
forecasting.py           # Monte Carlo simulation engine
tests/                   # Test suite
config.ini.example       # Configuration template
```
