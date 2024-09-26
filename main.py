import configparser
import logging
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from jira_data import JiraDataManager
from forecasting import MonteCarloForecaster
from jira_statistics import JiraStatistics, format_timedelta, print_statistics

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config

def display_individual_metrics(jira_manager: JiraDataManager, tickets: List[Dict]):
    contributors = set(ticket['assignee'] for ticket in tickets if ticket['assignee'])
    print("\nAvailable contributors:")
    for i, contributor in enumerate(contributors, 1):
        print(f"{i}. {contributor}")
    
    choice = input("Enter the number of the contributor to view detailed metrics (or 'q' to quit): ")
    if choice.lower() == 'q':
        return
    
    try:
        selected_contributor = list(contributors)[int(choice) - 1]
    except (ValueError, IndexError):
        print("Invalid selection. Returning to main menu.")
        return
    
    contributor_tickets = [t for t in tickets if t['assignee'] == selected_contributor]
    
    print(f"\nDetailed metrics for {selected_contributor}:")
    print(f"Total tickets: {len(contributor_tickets)}")
    print(f"Total story points: {sum(t['story_points'] for t in contributor_tickets if t['story_points'] is not None):.2f}")
    
    for status in set(status for t in contributor_tickets for status in t['cycle_times'].keys()):
        times = [t['cycle_times'][status] for t in contributor_tickets if status in t['cycle_times']]
        if times:
            avg_time = sum(times, timedelta()) / len(times)
            print(f"\n  Status: {status}")
            print(f"    Average time: {format_timedelta(avg_time)}")
            print(f"    Min time: {format_timedelta(min(times))} (Ticket: {min(contributor_tickets, key=lambda x: x['cycle_times'].get(status, timedelta.max))['key']})")
            print(f"    Max time: {format_timedelta(max(times))} (Ticket: {max(contributor_tickets, key=lambda x: x['cycle_times'].get(status, timedelta.min))['key']})")
    
    print("\nTicket IDs:")
    for ticket in contributor_tickets:
        print(f"  {ticket['key']}: {ticket['summary']} (Story Points: {ticket['story_points']})")

def prompt_for_statuses(jira_manager: JiraDataManager, tickets: List[Dict]) -> Set[str]:
    all_statuses = set()
    for ticket in tickets:
        all_statuses.update(ticket['cycle_times'].keys())
    
    print("Available statuses:")
    for i, status in enumerate(sorted(all_statuses), 1):
        print(f"{i}. {status}")
    
    selected_indices = input("Enter the numbers of the statuses you want to combine (comma-separated): ").split(',')
    selected_statuses = {sorted(all_statuses)[int(i.strip())-1] for i in selected_indices if i.strip().isdigit() and 0 < int(i.strip()) <= len(all_statuses)}
    
    return selected_statuses

def prompt_for_filter(jira_manager: JiraDataManager, prompt_text: str) -> str:
    while True:
        choice = input(f"{prompt_text}\nEnter a filter ID, or 'list' to see available filters: ").strip()
        if choice.lower() == "list":
            filters = jira_manager.get_filter_list()
            print("\nAvailable filters:")
            for i, (id, name) in enumerate(filters, 1):
                print(f"{i}. {name} (ID: {id})")
            print("\nNote: This list may not include all filters. You can still enter a filter ID directly.")
        elif choice.isdigit():
            return choice  # Return only the ID as a string
        else:
            print("Invalid input. Please enter a numeric filter ID or 'list'.")

def remove_outliers(tickets: List[Dict], iqr_multiplier: float = 1.5) -> Tuple[List[Dict], List[Dict]]:
    logger.info(f"Removing outliers using IQR method with multiplier {iqr_multiplier}")
    
    def calculate_iqr_bounds(data):
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (iqr_multiplier * iqr)
        upper_bound = q3 + (iqr_multiplier * iqr)
        return lower_bound, upper_bound

    # Calculate bounds for each status
    status_bounds = {}
    for ticket in tickets:
        for status, time in ticket['cycle_times'].items():
            if status not in status_bounds:
                status_bounds[status] = []
            status_bounds[status].append(time.total_seconds())

    for status, times in status_bounds.items():
        status_bounds[status] = calculate_iqr_bounds(times)

    # Filter out outliers
    filtered_tickets = []
    removed_outliers = []
    for ticket in tickets:
        is_outlier = False
        outlier_statuses = []
        for status, time in ticket['cycle_times'].items():
            lower_bound, upper_bound = status_bounds[status]
            if time.total_seconds() > upper_bound:  # Only remove long-running outliers
                is_outlier = True
                outlier_statuses.append(status)
        
        if is_outlier:
            removed_outliers.append(ticket)
            print(f"Removed outlier: Ticket {ticket['key']}")
            for status in outlier_statuses:
                print(f"  Status: {status}, Time: {ticket['cycle_times'][status]}, Upper bound: {timedelta(seconds=status_bounds[status][1])}")
        else:
            filtered_tickets.append(ticket)

    logger.info(f"Removed {len(removed_outliers)} outliers")
    return filtered_tickets, removed_outliers

def main():
    config = load_config()
    
    done_status = input("Enter the status to consider as 'Done' (default is 'Done'): ").strip() or "Done"
    jira_manager = JiraDataManager(config["Jira"]["email"], config["Jira"]["api_key"], done_status)
    jira_stats = JiraStatistics(jira_manager)
    forecaster = MonteCarloForecaster()

    historical_filter_id = prompt_for_filter(jira_manager, "Select the filter for historical data:")
    
    # Fetch tickets using the filter ID
    tickets = jira_manager.get_ticket_data(historical_filter_id)
    
    if not tickets:
        logger.error("No tickets found in the filter. Exiting.")
        return

    remove_outliers_choice = input("Do you want to remove outliers from the data? (y/n): ").lower() == 'y'
    if remove_outliers_choice:
        iqr_multiplier = float(input("Enter the IQR multiplier for outlier detection (default is 1.5): ") or 1.5)
        tickets, removed_outliers = remove_outliers(tickets, iqr_multiplier)
        print(f"\nRemoved {len(removed_outliers)} outliers.")
    
    # Optionally, you can save the removed outliers to a file for further analysis
   #  if removed_outliers:
        # outlier_file = "removed_outliers.json"
        # with open(outlier_file, "w") as f:
            # json.dump(removed_outliers, f, default=str, indent=2)
        # print(f"Removed outliers have been saved to {outlier_file}")
    
    tickets = jira_manager.backfill_story_points(tickets)
    
    combine_statuses = input("Do you want to see combined cycle time for specific statuses? (y/n): ").lower() == 'y'
    selected_statuses = prompt_for_statuses(jira_manager, tickets) if combine_statuses else None
    
    stats = jira_stats.get_filter_statistics(tickets, selected_statuses)
    print_statistics(stats)

    df_items, df_points = jira_manager.prepare_data(tickets)

    if df_items is None or df_points is None:
        logger.error("No data available for forecasting. Exiting.")
        return
    

    while True:
        print("\nChoose an analysis option:")
        print("1. Forecast completed work for various time periods")
        print("2. Forecast time to complete a specific amount of work")
        print("3. Forecast completion time for a backlog")
        print("4. Display detailed metrics for an individual")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ")

        if choice == "1":
            forecast_completed(forecaster, df_items, df_points)
        elif choice == "2":
            forecast_completion_time(forecaster, df_items, df_points)
        elif choice == "3":
            forecast_backlog_completion(jira_manager, forecaster, df_items, df_points)
        elif choice == "4":
            display_individual_metrics(jira_manager, tickets)
        elif choice == "5":
            logger.info("Exiting the program.")
            break
        else:
            logger.warning("Invalid choice. Please try again.")
            

def forecast_completed(forecaster: MonteCarloForecaster, df_items: pd.DataFrame, df_points: pd.DataFrame):
    forecast_points = input("Do you want to forecast using story points? (y/n): ").lower() == "y"
    df = df_points if forecast_points else df_items
    unit = "story points" if forecast_points else "items"
    
    print(f"\nProjected number of completed {unit}:")
    for days in [14, 28, 42, 56, 70, 84, 98, 112, 136]:
        projected = forecaster.run_simulation(df, num_days=days)
        sprint_count = int(days / 14)
        enddate = (datetime.now() + timedelta(days=days)).date()
        print(f"\n{sprint_count} sprints, ending {enddate} ({days} days from now):")
        print(f"  50th percentile: {projected[0]:.0f}")
        print(f"  75th percentile: {projected[1]:.0f}")
        print(f"  85th percentile: {projected[2]:.0f}")
        print(f"  95th percentile: {projected[3]:.0f}")

def forecast_completion_time(forecaster: MonteCarloForecaster, df_items: pd.DataFrame, df_points: pd.DataFrame):
    forecast_points = input("Do you want to forecast using story points? (y/n): ").lower() == "y"
    df = df_points if forecast_points else df_items
    unit = "story points" if forecast_points else "items"
    
    target = float(input(f"Enter the number of {unit} to complete: "))
    days_to_target = forecaster.run_simulation(df, target_count=target)
    
    print(f"\nEstimated days to complete {target:.0f} {unit}:")
    print(f"50% chance of completion within: {days_to_target[0]:.0f} days")
    print(f"75% chance of completion within: {days_to_target[1]:.0f} days")
    print(f"85% chance of completion within: {days_to_target[2]:.0f} days")
    print(f"95% chance of completion within: {days_to_target[3]:.0f} days")

def forecast_backlog_completion(jira_manager: JiraDataManager, forecaster: MonteCarloForecaster, df_items: pd.DataFrame, df_points: pd.DataFrame):
    unresolved_filter_id = prompt_for_filter(jira_manager, "Select the filter for unresolved items:")
    item_count, story_point_sum = jira_manager.get_unresolved_count(unresolved_filter_id)
    
    logger.info(f"Found {item_count} unresolved items totaling {story_point_sum:.2f} story points in the selected filter.")
    
    forecast_points = input("Do you want to forecast using story points? (y/n): ").lower() == "y"
    df = df_points if forecast_points else df_items
    unit = "story points" if forecast_points else "items"
    unresolved_count = storypoint_sum if forecast_points else item_count
    
    inflation = float(input(f"Enter the number of additional {unit} expected (default 0): ") or 0)
    total = unresolved_count + inflation

    logger.info(f"Forecasting completion time for {total:.2f} {unit}:")

    days_to_target, completion_dates = forecaster.forecast_backlog(df, total)

    print(f"\nProjected completion dates for {total:.2f} {unit}:")
    print(f"50% chance of completion by: {completion_dates[0]} (in {days_to_target[0]:.0f} days)")
    print(f"75% chance of completion by: {completion_dates[1]} (in {days_to_target[1]:.0f} days)")
    print(f"85% chance of completion by: {completion_dates[2]} (in {days_to_target[2]:.0f} days)")
    print(f"95% chance of completion by: {completion_dates[3]} (in {days_to_target[3]:.0f} days)")

if __name__ == "__main__":
    main()
