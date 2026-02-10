import pytest
import pandas as pd
from datetime import date, timedelta

from jira_data import JiraDataManager


@pytest.fixture
def manager():
    return JiraDataManager(
        email="test@example.com",
        api_key="fake-key",
        base_url="https://example.atlassian.net",
    )


def _make_processed_ticket(
    key="TEST-1",
    story_points=3,
    completed_date=date(2024, 6, 15),
    cycle_times=None,
):
    if cycle_times is None:
        cycle_times = {"In Progress": timedelta(days=2)}
    return {
        "key": key,
        "story_points": story_points,
        "summary": f"Summary for {key}",
        "created_date": date(2024, 6, 1),
        "completed_date": completed_date,
        "assignee": "Alice",
        "description_length": 100,
        "acceptance_criteria": "Some criteria",
        "acceptance_criteria_length": 13,
        "cycle_times": cycle_times,
        "backlog_to_progress": 1,
    }


class TestBackfillStoryPoints:
    def test_fills_zero_points_with_average(self, manager):
        tickets = [
            _make_processed_ticket(key="T-1", story_points=4),
            _make_processed_ticket(key="T-2", story_points=6),
            _make_processed_ticket(key="T-3", story_points=0),
        ]
        result = manager.backfill_story_points(tickets)
        assert result[2]["story_points"] == 5.0  # average of 4 and 6
        assert result[2]["original_story_points"] is False

    def test_fills_none_points_with_average(self, manager):
        tickets = [
            _make_processed_ticket(key="T-1", story_points=2),
            _make_processed_ticket(key="T-2", story_points=None),
        ]
        result = manager.backfill_story_points(tickets)
        assert result[1]["story_points"] == 2.0
        assert result[1]["original_story_points"] is False

    def test_marks_original_points(self, manager):
        tickets = [_make_processed_ticket(key="T-1", story_points=3)]
        result = manager.backfill_story_points(tickets)
        assert result[0]["original_story_points"] is True

    def test_all_zero_uses_default(self, manager):
        tickets = [
            _make_processed_ticket(key="T-1", story_points=0),
            _make_processed_ticket(key="T-2", story_points=0),
        ]
        result = manager.backfill_story_points(tickets)
        assert result[0]["story_points"] == 1  # default when no valid points


class TestPrepareData:
    def test_returns_two_dataframes(self, manager):
        tickets = [
            _make_processed_ticket(key="T-1", story_points=3, completed_date=date(2024, 6, 10)),
            _make_processed_ticket(key="T-2", story_points=5, completed_date=date(2024, 6, 12)),
        ]
        df_items, df_points = manager.prepare_data(tickets)
        assert isinstance(df_items, pd.DataFrame)
        assert isinstance(df_points, pd.DataFrame)

    def test_has_expected_columns(self, manager):
        tickets = [
            _make_processed_ticket(completed_date=date(2024, 6, 10)),
        ]
        df_items, df_points = manager.prepare_data(tickets)
        assert "Completed" in df_items.columns
        assert "Cumulative Completed" in df_items.columns
        assert "Completed" in df_points.columns
        assert "Cumulative Completed" in df_points.columns

    def test_item_counts_correct(self, manager):
        tickets = [
            _make_processed_ticket(key="T-1", completed_date=date(2024, 6, 10)),
            _make_processed_ticket(key="T-2", completed_date=date(2024, 6, 10)),
            _make_processed_ticket(key="T-3", completed_date=date(2024, 6, 12)),
        ]
        df_items, _ = manager.prepare_data(tickets)
        day1 = pd.Timestamp("2024-06-10")
        day3 = pd.Timestamp("2024-06-12")
        assert df_items.at[day1, "Completed"] == 2
        assert df_items.at[day3, "Completed"] == 1

    def test_story_points_summed(self, manager):
        tickets = [
            _make_processed_ticket(key="T-1", story_points=3, completed_date=date(2024, 6, 10)),
            _make_processed_ticket(key="T-2", story_points=5, completed_date=date(2024, 6, 10)),
        ]
        _, df_points = manager.prepare_data(tickets)
        day1 = pd.Timestamp("2024-06-10")
        assert df_points.at[day1, "Completed"] == 8

    def test_returns_none_for_no_closed_tickets(self, manager):
        tickets = [_make_processed_ticket(completed_date=None)]
        result = manager.prepare_data(tickets)
        assert result == (None, None)

    def test_cumulative_sums(self, manager):
        tickets = [
            _make_processed_ticket(key="T-1", completed_date=date(2024, 6, 10)),
            _make_processed_ticket(key="T-2", completed_date=date(2024, 6, 11)),
            _make_processed_ticket(key="T-3", completed_date=date(2024, 6, 12)),
        ]
        df_items, _ = manager.prepare_data(tickets)
        day3 = pd.Timestamp("2024-06-12")
        assert df_items.at[day3, "Cumulative Completed"] == 3


class TestCalculateCycleTimes:
    def test_simple_transition(self, manager):
        changelog = [
            {
                "created": "2024-06-10T10:00:00.000+0000",
                "items": [
                    {
                        "field": "status",
                        "fromString": "To Do",
                        "toString": "In Progress",
                    }
                ],
            },
            {
                "created": "2024-06-12T10:00:00.000+0000",
                "items": [
                    {
                        "field": "status",
                        "fromString": "In Progress",
                        "toString": "Done",
                    }
                ],
            },
        ]
        cycle_times, backlog_count = manager.calculate_cycle_times(changelog)
        assert "In Progress" in cycle_times
        assert cycle_times["In Progress"].days == 2
        assert backlog_count == 0

    def test_backlog_to_progress_counted(self, manager):
        changelog = [
            {
                "created": "2024-06-10T10:00:00.000+0000",
                "items": [
                    {
                        "field": "status",
                        "fromString": "Backlog",
                        "toString": "In Progress",
                    }
                ],
            },
        ]
        _, backlog_count = manager.calculate_cycle_times(changelog)
        assert backlog_count == 1

    def test_non_status_fields_ignored(self, manager):
        changelog = [
            {
                "created": "2024-06-10T10:00:00.000+0000",
                "items": [
                    {
                        "field": "assignee",
                        "fromString": "Alice",
                        "toString": "Bob",
                    }
                ],
            },
        ]
        cycle_times, backlog_count = manager.calculate_cycle_times(changelog)
        assert len(cycle_times) == 0
        assert backlog_count == 0


class TestProcessIssues:
    def test_processes_single_issue(self, manager):
        issues = [
            {
                "key": "TEST-1",
                "fields": {
                    manager.STORY_POINTS_FIELD: 5,
                    manager.ACCEPTANCE_CRITERIA_FIELD: "AC text",
                    "summary": "Test ticket",
                    "created": "2024-06-01T10:00:00.000+0000",
                    "resolutiondate": "2024-06-15T10:00:00.000+0000",
                    "assignee": {"displayName": "Alice"},
                    "description": "Description text",
                    "status": {"name": "Done"},
                },
                "changelog": {"histories": []},
            }
        ]
        result = manager.process_issues(issues)
        assert len(result) == 1
        assert result[0]["key"] == "TEST-1"
        assert result[0]["story_points"] == 5
        assert result[0]["assignee"] == "Alice"

    def test_handles_null_fields(self, manager):
        issues = [
            {
                "key": "TEST-2",
                "fields": {
                    manager.STORY_POINTS_FIELD: None,
                    manager.ACCEPTANCE_CRITERIA_FIELD: None,
                    "summary": "No points",
                    "created": "2024-06-01T10:00:00.000+0000",
                    "resolutiondate": None,
                    "assignee": None,
                    "description": None,
                    "status": {"name": "Open"},
                },
                "changelog": {"histories": []},
            }
        ]
        result = manager.process_issues(issues)
        assert result[0]["story_points"] == 0
        assert result[0]["assignee"] is None
        assert result[0]["completed_date"] is None
