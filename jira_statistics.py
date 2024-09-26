import logging
from typing import List, Dict, Set
from collections import defaultdict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class JiraStatistics:
    def __init__(self, jira_manager):
        self.jira_manager = jira_manager

    def get_filter_statistics(
        self, tickets: List[Dict], selected_statuses: Set[str] = None
    ) -> Dict:
        stats = {
            "cycle_time": self.calculate_cycle_time_stats(tickets),
            "combined_cycle_time": (
                self.calculate_combined_cycle_time_stats(tickets, selected_statuses)
                if selected_statuses
                else None
            ),
            "contributors": self.analyze_contributors(tickets, selected_statuses),
            "ticket_range": self.get_ticket_range(tickets),
            "story_points": self.analyze_story_points(tickets),
            "correlations": self.get_correlations(tickets),
            "backlog_to_progress": self.analyze_backlog_to_progress(tickets),
        }

        logger.debug(f"Generated statistics: {stats}")

        return stats

    def calculate_combined_cycle_time_stats(
        self, tickets: List[Dict], selected_statuses: Set[str]
    ) -> Dict:
        logger.info(
            f"Calculating combined cycle time statistics for statuses: {', '.join(selected_statuses)}"
        )
        combined_times = []

        for ticket in tickets:
            combined_time = self.jira_manager.calculate_combined_cycle_time(
                ticket["cycle_times"], selected_statuses
            )
            combined_times.append(combined_time.total_seconds())

        return {
            "average": timedelta(seconds=np.mean(combined_times)),
            "median": timedelta(seconds=np.median(combined_times)),
            "std_dev": timedelta(seconds=np.std(combined_times)),
            "min": timedelta(seconds=np.min(combined_times)),
            "max": timedelta(seconds=np.max(combined_times)),
        }

    def calculate_cycle_time_stats(self, tickets: List[Dict]) -> Dict:
        logger.info("Calculating cycle time statistics")
        all_cycle_times = defaultdict(list)

        for ticket in tickets:
            for status, time in ticket["cycle_times"].items():
                all_cycle_times[status].append((time.total_seconds(), ticket["key"]))

        cycle_time_stats = {}
        for status, times in all_cycle_times.items():
            times_seconds = [t[0] for t in times]
            cycle_time_stats[status] = {
                "average": timedelta(seconds=np.mean(times_seconds)),
                "median": timedelta(seconds=np.median(times_seconds)),
                "std_dev": timedelta(seconds=np.std(times_seconds)),
                "min": timedelta(seconds=min(times_seconds)),
                "min_ticket": min(times, key=lambda x: x[0])[1],
                "max": timedelta(seconds=max(times_seconds)),
                "max_ticket": max(times, key=lambda x: x[0])[1],
            }

        return cycle_time_stats

    def get_ticket_range(self, tickets: List[Dict]) -> Dict:
        logger.info("Determining first and last completed tickets")
        completed_tickets = [t for t in tickets if t["completed_date"]]

        if not completed_tickets:
            logger.warning("No completed tickets found.")
            return {}

        first_ticket = min(completed_tickets, key=lambda x: x["completed_date"])
        last_ticket = max(completed_tickets, key=lambda x: x["completed_date"])

        return {
            "first_ticket": {
                "key": first_ticket["key"],
                "completion_date": first_ticket["completed_date"],
            },
            "last_ticket": {
                "key": last_ticket["key"],
                "completion_date": last_ticket["completed_date"],
            },
        }

    def analyze_backlog_to_progress(self, tickets: List[Dict]) -> Dict:
        logger.info("Analyzing backlog to progress transitions")
        transitions = [ticket["backlog_to_progress"] for ticket in tickets]
        return {
            "total_transitions": sum(transitions),
            "tickets_with_transitions": len([t for t in transitions if t > 0]),
            "max_transitions": max(transitions),
            "average_transitions": np.mean(transitions),
        }

    def analyze_contributors(
        self, tickets: List[Dict], selected_statuses: Set[str] = None
    ) -> Dict:
        logger.info("Analyzing contributor statistics")
        contributors = defaultdict(
            lambda: {
                "count": 0,
                "points": 0,
                "cycle_times": defaultdict(list),
                "first_completion": None,
                "last_completion": None,
            }
        )

        for ticket in tickets:
            if ticket["assignee"] and ticket["completed_date"]:
                assignee = ticket["assignee"]
                contributors[assignee]["count"] += 1
                contributors[assignee]["points"] += ticket["story_points"]

                for status, time in ticket["cycle_times"].items():
                    contributors[assignee]["cycle_times"][status].append(time)

                completion_date = ticket["completed_date"]
                if (
                    contributors[assignee]["first_completion"] is None
                    or completion_date < contributors[assignee]["first_completion"]
                ):
                    contributors[assignee]["first_completion"] = completion_date
                if (
                    contributors[assignee]["last_completion"] is None
                    or completion_date > contributors[assignee]["last_completion"]
                ):
                    contributors[assignee]["last_completion"] = completion_date
                if selected_statuses:
                    combined_time = self.jira_manager.calculate_combined_cycle_time(
                        ticket["cycle_times"], selected_statuses
                    )
                    contributors[assignee]["combined_cycle_times"].append(combined_time)

        for assignee, data in contributors.items():
            data["avg_cycle_times"] = {
                status: sum(times, timedelta()) / len(times) if times else timedelta(0)
                for status, times in data["cycle_times"].items()
            }
            if selected_statuses:
                data["avg_combined_cycle_time"] = (
                    sum(data["combined_cycle_times"], timedelta())
                    / len(data["combined_cycle_times"])
                    if data["combined_cycle_times"]
                    else timedelta(0)
                )

        return {"count": len(contributors), "details": dict(contributors)}

    def get_ticket_range(self, tickets: List[Dict]) -> Dict:
        logger.info("Determining first and last completed tickets")
        completed_tickets = [t for t in tickets if t["completed_date"]]

        if not completed_tickets:
            logger.warning("No completed tickets found.")
            return {}

        first_ticket = min(completed_tickets, key=lambda x: x["completed_date"])
        last_ticket = max(completed_tickets, key=lambda x: x["completed_date"])

        return {
            "first_ticket": {
                "key": first_ticket["key"],
                "completion_date": first_ticket["completed_date"],
            },
            "last_ticket": {
                "key": last_ticket["key"],
                "completion_date": last_ticket["completed_date"],
            },
        }

    def analyze_story_points(self, tickets: List[Dict]) -> Dict:
        logger.info("Analyzing story point data")

        for i, ticket in enumerate(tickets):
            logger.debug(
                f"Ticket {i+1}: Key: {ticket.get('key')}, Story Points: {ticket.get('story_points')}, Original: {ticket.get('original_story_points')}"
            )

        original_tickets = [
            t for t in tickets if t.get("original_story_points") is True
        ]
        backfilled_tickets = [
            t for t in tickets if t.get("original_story_points") is False
        ]

        logger.info(f"Original tickets: {len(original_tickets)}")
        logger.info(f"Backfilled tickets: {len(backfilled_tickets)}")

        total_original_points = sum(t["story_points"] for t in original_tickets)
        total_backfilled_points = sum(t["story_points"] for t in backfilled_tickets)
        total_all_points = total_original_points + total_backfilled_points

        logger.info(f"Total original points: {total_original_points}")
        logger.info(f"Total backfilled points: {total_backfilled_points}")
        logger.info(f"Total all points: {total_all_points}")

        if original_tickets:
            average_points = np.mean([t["story_points"] for t in original_tickets])
        else:
            average_points = None

        logger.info(f"Average points: {average_points}")

        return {
            "tickets_with_points": len(original_tickets),
            "backfilled_tickets": len(backfilled_tickets),
            "average_points": average_points,
            "total_original_points": total_original_points,
            "total_backfilled_points": total_backfilled_points,
            "total_all_points": total_all_points,
            "backfilled_points": len(backfilled_tickets) > 0,
        }

    def get_correlations(self, tickets: List[Dict]) -> Dict:
        logger.info("Calculating correlations")
        df = pd.DataFrame(tickets)

        correlations = {}

        def timedelta_to_seconds(td):
            return td.total_seconds() if pd.notnull(td) else pd.NaT

        if "acceptance_criteria_length" in df.columns and "cycle_time" in df.columns:
            cycle_times = df["cycle_time"].apply(timedelta_to_seconds)
            correlations["acceptance_criteria_cycle_time"] = df[
                "acceptance_criteria_length"
            ].corr(cycle_times)

        if "description_length" in df.columns and "cycle_time" in df.columns:
            cycle_times = df["cycle_time"].apply(timedelta_to_seconds)
            correlations["description_cycle_time"] = df["description_length"].corr(
                cycle_times
            )

        if "story_points" in df.columns and "cycle_time" in df.columns:
            df_with_points = df.dropna(subset=["story_points", "cycle_time"])
            cycle_times = df_with_points["cycle_time"].apply(timedelta_to_seconds)
            correlations["story_points_cycle_time"] = df_with_points[
                "story_points"
            ].corr(cycle_times)

        return correlations


def format_timedelta(td: timedelta) -> str:
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{days}d {hours}h {minutes}m"


def print_statistics(stats: Dict):
    print("\n=== Jira Filter Statistics ===\n")

    # Cycle Time Statistics
    if "cycle_time" in stats and stats["cycle_time"]:
        print("Cycle Time Statistics:")
        for status, cycle_stats in stats["cycle_time"].items():
            print(f"  Status: {status}")
            print(f"    Average: {format_timedelta(cycle_stats['average'])}")
            print(f"    Median: {format_timedelta(cycle_stats['median'])}")
            print(f"    Standard Deviation: {format_timedelta(cycle_stats['std_dev'])}")
            print(
                f"    Minimum: {format_timedelta(cycle_stats['min'])} (Ticket: {cycle_stats['min_ticket']})"
            )
            print(
                f"    Maximum: {format_timedelta(cycle_stats['max'])} (Ticket: {cycle_stats['max_ticket']})"
            )
    else:
        print("No cycle time data available.")

    # Combined Cycle Time Statistics
    if "combined_cycle_time" in stats and stats["combined_cycle_time"]:
        print("\nCombined Cycle Time Statistics:")
        combined_stats = stats["combined_cycle_time"]
        print(f"  Average: {format_timedelta(combined_stats['average'])}")
        print(f"  Median: {format_timedelta(combined_stats['median'])}")
        print(f"  Standard Deviation: {format_timedelta(combined_stats['std_dev'])}")
        print(f"  Minimum: {format_timedelta(combined_stats['min'])}")
        print(f"  Maximum: {format_timedelta(combined_stats['max'])}")

    # Contributors
    if "contributors" in stats and stats["contributors"]:
        print("\nContributor Statistics:")
        print(f"  Total Contributors: {stats['contributors']['count']}")
        for assignee, data in stats["contributors"]["details"].items():
            print(f"  {assignee}:")
            print(f"    Tickets Completed: {data['count']}")
            print(f"    Story Points Completed: {data['points']:.2f}")
            print("    Average Cycle Times:")
            for status, avg_time in data["avg_cycle_times"].items():
                print(f"      {status}: {format_timedelta(avg_time)}")
            if "avg_combined_cycle_time" in data:
                print(
                    f"    Average Combined Cycle Time: {format_timedelta(data['avg_combined_cycle_time'])}"
                )
            print(f"    First Completion: {data['first_completion']}")
            print(f"    Last Completion: {data['last_completion']}")
    else:
        print("\nNo contributor data available.")

    # Ticket Range
    if "ticket_range" in stats and stats["ticket_range"]:
        print("\nTicket Completion Range:")
        print(
            f"  First Ticket: {stats['ticket_range']['first_ticket']['key']} (Completed: {stats['ticket_range']['first_ticket']['completion_date']})"
        )
        print(
            f"  Last Ticket: {stats['ticket_range']['last_ticket']['key']} (Completed: {stats['ticket_range']['last_ticket']['completion_date']})"
        )
    else:
        print("\nNo ticket range data available.")

    # Story Points
    if "story_points" in stats and stats["story_points"]:
        print("\nStory Point Statistics:")
        sp_stats = stats["story_points"]
        print(
            f"  Tickets with Original Story Points: {sp_stats['tickets_with_points']}"
        )
        print(
            f"  Tickets with Backfilled Story Points: {sp_stats['backfilled_tickets']}"
        )
        if sp_stats["average_points"]:
            print(
                f"  Average Story Points (from original data): {sp_stats['average_points']:.2f}"
            )
        else:
            print("  No original story point data available.")

        print(
            f"  Total Story Points (original): {sp_stats['total_original_points']:.2f}"
        )
        print(
            f"  Total Story Points (backfilled): {sp_stats['total_backfilled_points']:.2f}"
        )
        print(f"  Total Story Points (all): {sp_stats['total_all_points']:.2f}")

        if sp_stats["backfilled_points"]:
            print(
                "  Note: Some tickets had their story points backfilled with the average for statistical analysis."
            )
    else:
        print("\nNo story point data available.")

    # Correlations
    if "correlations" in stats and stats["correlations"]:
        print("\nCorrelations:")
        if "acceptance_criteria_cycle_time" in stats["correlations"]:
            print(
                f"  Acceptance Criteria Length vs Cycle Time: {stats['correlations']['acceptance_criteria_cycle_time']:.2f}"
            )
        if "description_cycle_time" in stats["correlations"]:
            print(
                f"  Description Length vs Cycle Time: {stats['correlations']['description_cycle_time']:.2f}"
            )
        if "story_points_cycle_time" in stats["correlations"]:
            print(
                f"  Story Points vs Cycle Time: {stats['correlations']['story_points_cycle_time']:.2f}"
            )
    else:
        print("\nNo correlation data available.")

    # Backlog to Progress Transitions
    if "backlog_to_progress" in stats and stats["backlog_to_progress"]:
        print("\nBacklog to Progress Transition Statistics:")
        btp_stats = stats["backlog_to_progress"]
        print(f"  Total Transitions: {btp_stats['total_transitions']}")
        print(f"  Tickets with Transitions: {btp_stats['tickets_with_transitions']}")
        print(f"  Max Transitions for a Single Ticket: {btp_stats['max_transitions']}")
        print(
            f"  Average Transitions per Ticket: {btp_stats['average_transitions']:.2f}"
        )
    else:
        print("\nNo backlog to progress transition data available.")
