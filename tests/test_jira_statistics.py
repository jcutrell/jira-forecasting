import pytest
from datetime import date, timedelta
from unittest.mock import MagicMock

from jira_statistics import JiraStatistics, format_timedelta


def _make_ticket(
    key="TEST-1",
    assignee="Alice",
    story_points=3,
    completed_date=date(2024, 6, 15),
    cycle_times=None,
    backlog_to_progress=0,
    original_story_points=True,
    acceptance_criteria_length=50,
    description_length=100,
):
    if cycle_times is None:
        cycle_times = {
            "In Progress": timedelta(days=2),
            "In Review": timedelta(days=1),
        }
    return {
        "key": key,
        "assignee": assignee,
        "story_points": story_points,
        "completed_date": completed_date,
        "cycle_times": cycle_times,
        "backlog_to_progress": backlog_to_progress,
        "original_story_points": original_story_points,
        "acceptance_criteria_length": acceptance_criteria_length,
        "description_length": description_length,
    }


@pytest.fixture
def stats():
    mock_manager = MagicMock()
    return JiraStatistics(mock_manager)


@pytest.fixture
def sample_tickets():
    return [
        _make_ticket(
            key="TEST-1",
            assignee="Alice",
            story_points=3,
            completed_date=date(2024, 6, 10),
            cycle_times={
                "In Progress": timedelta(days=2),
                "In Review": timedelta(days=1),
            },
            backlog_to_progress=1,
        ),
        _make_ticket(
            key="TEST-2",
            assignee="Bob",
            story_points=5,
            completed_date=date(2024, 6, 15),
            cycle_times={
                "In Progress": timedelta(days=4),
                "In Review": timedelta(days=2),
            },
            backlog_to_progress=2,
        ),
        _make_ticket(
            key="TEST-3",
            assignee="Alice",
            story_points=2,
            completed_date=date(2024, 6, 20),
            cycle_times={
                "In Progress": timedelta(days=1),
                "In Review": timedelta(hours=12),
            },
            backlog_to_progress=0,
        ),
    ]


class TestCalculateCombinedCycleTime:
    def test_sums_selected_statuses(self):
        cycle_times = {
            "In Progress": timedelta(days=2),
            "In Review": timedelta(days=1),
            "Done": timedelta(days=0),
        }
        result = JiraStatistics.calculate_combined_cycle_time(
            cycle_times, {"In Progress", "In Review"}
        )
        assert result == timedelta(days=3)

    def test_missing_status_treated_as_zero(self):
        cycle_times = {"In Progress": timedelta(days=2)}
        result = JiraStatistics.calculate_combined_cycle_time(
            cycle_times, {"In Progress", "In Review"}
        )
        assert result == timedelta(days=2)

    def test_empty_statuses(self):
        cycle_times = {"In Progress": timedelta(days=2)}
        result = JiraStatistics.calculate_combined_cycle_time(cycle_times, set())
        assert result == timedelta(0)


class TestCalculateCycleTimeStats:
    def test_returns_stats_per_status(self, stats, sample_tickets):
        result = stats.calculate_cycle_time_stats(sample_tickets)
        assert "In Progress" in result
        assert "In Review" in result

    def test_stats_have_expected_keys(self, stats, sample_tickets):
        result = stats.calculate_cycle_time_stats(sample_tickets)
        for status_stats in result.values():
            assert "average" in status_stats
            assert "median" in status_stats
            assert "std_dev" in status_stats
            assert "min" in status_stats
            assert "min_ticket" in status_stats
            assert "max" in status_stats
            assert "max_ticket" in status_stats

    def test_min_max_tickets_identified(self, stats, sample_tickets):
        result = stats.calculate_cycle_time_stats(sample_tickets)
        # TEST-2 had 4 days In Progress (max), TEST-3 had 1 day (min)
        assert result["In Progress"]["max_ticket"] == "TEST-2"
        assert result["In Progress"]["min_ticket"] == "TEST-3"


class TestGetTicketRange:
    def test_returns_first_and_last(self, stats, sample_tickets):
        result = stats.get_ticket_range(sample_tickets)
        assert result["first_ticket"]["key"] == "TEST-1"
        assert result["first_ticket"]["completion_date"] == date(2024, 6, 10)
        assert result["last_ticket"]["key"] == "TEST-3"
        assert result["last_ticket"]["completion_date"] == date(2024, 6, 20)

    def test_empty_when_no_completed(self, stats):
        tickets = [_make_ticket(completed_date=None)]
        result = stats.get_ticket_range(tickets)
        assert result == {}


class TestAnalyzeBacklogToProgress:
    def test_counts_transitions(self, stats, sample_tickets):
        result = stats.analyze_backlog_to_progress(sample_tickets)
        assert result["total_transitions"] == 3  # 1 + 2 + 0
        assert result["tickets_with_transitions"] == 2
        assert result["max_transitions"] == 2


class TestAnalyzeContributors:
    def test_counts_contributors(self, stats, sample_tickets):
        result = stats.analyze_contributors(sample_tickets)
        assert result["count"] == 2  # Alice and Bob

    def test_contributor_ticket_counts(self, stats, sample_tickets):
        result = stats.analyze_contributors(sample_tickets)
        assert result["details"]["Alice"]["count"] == 2
        assert result["details"]["Bob"]["count"] == 1

    def test_contributor_points(self, stats, sample_tickets):
        result = stats.analyze_contributors(sample_tickets)
        assert result["details"]["Alice"]["points"] == 5  # 3 + 2
        assert result["details"]["Bob"]["points"] == 5

    def test_with_selected_statuses(self, stats, sample_tickets):
        result = stats.analyze_contributors(
            sample_tickets, selected_statuses={"In Progress", "In Review"}
        )
        alice = result["details"]["Alice"]
        assert "avg_combined_cycle_time" in alice
        assert len(alice["combined_cycle_times"]) == 2

    def test_skips_unassigned_tickets(self, stats):
        tickets = [_make_ticket(assignee=None)]
        result = stats.analyze_contributors(tickets)
        assert result["count"] == 0


class TestAnalyzeStoryPoints:
    def test_separates_original_and_backfilled(self, stats):
        tickets = [
            _make_ticket(story_points=3, original_story_points=True),
            _make_ticket(story_points=2, original_story_points=False),
        ]
        result = stats.analyze_story_points(tickets)
        assert result["tickets_with_points"] == 1
        assert result["backfilled_tickets"] == 1
        assert result["total_original_points"] == 3
        assert result["total_backfilled_points"] == 2
        assert result["total_all_points"] == 5

    def test_no_original_points(self, stats):
        tickets = [_make_ticket(story_points=2, original_story_points=False)]
        result = stats.analyze_story_points(tickets)
        assert result["average_points"] is None


class TestGetCorrelations:
    def test_returns_correlations_with_enough_data(self, stats, sample_tickets):
        result = stats.get_correlations(sample_tickets)
        assert "acceptance_criteria_cycle_time" in result
        assert "description_cycle_time" in result

    def test_empty_with_single_ticket(self, stats):
        result = stats.get_correlations([_make_ticket()])
        assert result == {}

    def test_story_points_correlation_present(self, stats, sample_tickets):
        result = stats.get_correlations(sample_tickets)
        assert "story_points_cycle_time" in result


class TestFormatTimedelta:
    def test_simple_days(self):
        assert format_timedelta(timedelta(days=3, hours=2, minutes=15)) == "3d 2h 15m"

    def test_zero(self):
        assert format_timedelta(timedelta(0)) == "0d 0h 0m"

    def test_hours_only(self):
        assert format_timedelta(timedelta(hours=5, minutes=30)) == "0d 5h 30m"


class TestGetFilterStatistics:
    def test_returns_all_keys(self, stats, sample_tickets):
        result = stats.get_filter_statistics(sample_tickets)
        assert "cycle_time" in result
        assert "combined_cycle_time" in result
        assert "contributors" in result
        assert "ticket_range" in result
        assert "story_points" in result
        assert "correlations" in result
        assert "backlog_to_progress" in result

    def test_combined_cycle_time_none_without_statuses(self, stats, sample_tickets):
        result = stats.get_filter_statistics(sample_tickets)
        assert result["combined_cycle_time"] is None

    def test_combined_cycle_time_present_with_statuses(self, stats, sample_tickets):
        result = stats.get_filter_statistics(
            sample_tickets, selected_statuses={"In Progress"}
        )
        assert result["combined_cycle_time"] is not None
        assert "average" in result["combined_cycle_time"]
