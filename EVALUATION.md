# Project Structure Evaluation

## Overview

This is a ~1,100-line Python CLI tool for Jira workflow forecasting using Monte Carlo simulation. It has 5 source files with clear separation of concerns. This document catalogues structural issues and recommended improvements.

## What works well

- **Modular design**: Each file has a distinct role (API, statistics, forecasting, visualization, CLI).
- **Type hints**: Used consistently across the codebase.
- **Logging**: Set up throughout with appropriate levels.
- **Error handling**: Present around API calls with try/except and status code checks.

## Bugs

### 1. Duplicate `get_ticket_range` method (jira_statistics.py:81 and :168)

The method is defined twice in `JiraStatistics`. Python silently uses the last definition. The first is dead code.

### 2. Duplicate and broken `get_correlations` (jira_data.py:294 and jira_statistics.py:233)

Both implementations reference a `cycle_time` column that doesn't exist in the ticket dicts (the actual key is `cycle_times`, a dict of timedeltas per status). Correlations always return `{}`.

### 3. Missing `combined_cycle_times` key (jira_statistics.py:151)

The `contributors` defaultdict initializes with `cycle_times` but not `combined_cycle_times`. When `selected_statuses` is provided, `data["combined_cycle_times"]` raises a `KeyError`.

### 4. Broken visualization column references (visualization_utils.py)

- `plot_cumulative_flow` references `"Cumulative Completed Items"` but `prepare_data` creates `"Cumulative Completed"`.
- `plot_throughput_trend` references `"Completed Items"` but the actual column is `"Completed"`.

## Missing infrastructure

### 5. No tests

Zero test files. The Monte Carlo simulation engine and statistics calculations have no correctness verification.

### 6. No `pyproject.toml` or `setup.py`

The project cannot be installed as a package. No Python version is specified.

### 7. Unpinned dependencies

`requirements.txt` lists packages without versions. `scipy` is listed but never imported anywhere.

### 8. No CI/CD

No GitHub Actions workflow or automated checks of any kind.

## Code quality issues

### 9. Hardcoded Jira custom field IDs

`customfield_10026` and `customfield_10082` are scattered across `jira_data.py`. These should be configurable.

### 10. Duplicate `load_config()`

Defined in both `main.py:18` and `jira_data.py:15`. `JiraDataManager.__init__` reads config internally despite receiving values from config as constructor args.

### 11. Inconsistent config handling in `JiraDataManager`

`email` and `api_key` are passed as parameters, but `base_url` is read from config internally. Should be consistent.

### 12. Mixed Jira API versions

`get_filter_list`/`get_account_id` use `/rest/api/3/` while `get_ticket_data` uses `/rest/api/2/`.

### 13. Dead code in `forecasting.py`

`forecast_completed_items` and `forecast_completion_time` (lines 35-41) are never called.

### 14. No CLI input validation

`float(input(...))` in `main.py` will crash on non-numeric input.

## Project hygiene

### 15. No README

No setup instructions, config.ini format documentation, or usage guide.

### 16. `scratch.md` committed to repo

Personal notes file should be gitignored.

### 17. Inconsistent indentation in `jira_data.py:118-159`

`calculate_cycle_times` uses 8-space indentation vs. 4-space everywhere else.

### 18. Commented-out code in `main.py:154-158`

Dead outlier-saving code should be removed or made into a feature.

### 19. No linting or formatting config

No `.flake8`, `ruff.toml`, black config, or pre-commit hooks.

## Resolution status

| # | Issue | Status |
|---|-------|--------|
| 1 | Duplicate `get_ticket_range` | **Fixed** -- removed duplicate |
| 2 | Duplicate/broken `get_correlations` | **Fixed** -- rewrote to use `cycle_times` dict, removed duplicate from jira_data.py |
| 3 | Missing `combined_cycle_times` key + missing `calculate_combined_cycle_time` method | **Fixed** -- added static method to JiraStatistics, added key to defaultdict |
| 4 | Broken visualization column names | **Fixed** -- matched to `prepare_data` output |
| 5 | No tests | **Fixed** -- 50 tests across 3 test files |
| 6 | No `pyproject.toml` | **Fixed** -- added with pinned deps, Python >=3.9, pytest/ruff dev deps |
| 7 | Unpinned dependencies | **Fixed** -- pinned in requirements.txt, removed unused scipy, added missing pytz |
| 8 | No CI/CD | **Not done** -- deferred until tests are verified on a real Jira instance |
| 9 | Hardcoded custom field IDs | **Fixed** -- extracted to class constants on JiraDataManager |
| 10 | Duplicate `load_config()` | **Fixed** -- consolidated, base_url now passed explicitly |
| 11 | Inconsistent config handling | **Fixed** -- all config values passed as constructor params |
| 12 | Mixed Jira API versions | **Not done** -- risky without a test Jira instance |
| 13 | Dead code in forecasting.py | **Fixed** -- removed unused wrapper methods |
| 14 | No CLI input validation | **Fixed** -- try/except around float(input()) calls |
| 15 | No README | **Fixed** -- added with setup, usage, and project structure |
| 16 | scratch.md in repo | **Fixed** -- deleted |
| 17 | Inconsistent indentation | **Fixed** -- normalized to 4-space |
| 18 | Commented-out code | **Fixed** -- removed |
| 19 | No linting config | **Fixed** -- ruff configured, codebase formatted, all checks pass |
