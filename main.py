import configparser
import logging
from typing import List, Dict
from datetime import datetime, timedelta
import pandas as pd

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

def prompt_for_filter(jira_manager: JiraDataManager, prompt_text: str) -> str:
    while True:
        choice = input(f"{prompt_text}\nEnter a filter ID, or 'list' to see available filters: ").strip()
        if choice.lower() == "list":
            filters = jira_manager.get_filter_list()
            print("\nAvailable filters:")
            for i, (id, name) in enumerate(filters):
                print(f"{i+1}. {name} (ID: {id})")
            print("\nNote: This list may not include all filters. You can still enter a filter ID directly.")
        elif choice.isdigit():
            return choice
        else:
            print("Invalid input. Please enter a numeric filter ID or 'list'.")

def main():
    config = load_config()
    
    done_status = input("Enter the status to consider as 'Done' (default is 'Done'): ").strip() or "Done"
    jira_manager = JiraDataManager(config["Jira"]["email"], config["Jira"]["api_key"], done_status)
    jira_stats = JiraStatistics(jira_manager)
    forecaster = MonteCarloForecaster()

    historical_filter_id = prompt_for_filter(jira_manager, "Select the filter for historical data:")
    
    use_story_points = input("Use story points for calculations? (y/n): ").lower() == 'y'
    
    stats = jira_stats.get_filter_statistics(historical_filter_id, use_story_points)
    print_statistics(stats, use_story_points)

    df, _ = jira_manager.prepare_data(historical_filter_id, use_story_points)

    if df is None:
        logger.error("No data available for forecasting. Exiting.")
        return

    while True:
        print("\nChoose an analysis option:")
        print("1. Forecast completed items/points for various time periods")
        print("2. Forecast time to complete a specific number of items/points")
        print("3. Forecast completion time for a backlog")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        if choice == "1":
            forecast_completed(forecaster, df, use_story_points)
        elif choice == "2":
            forecast_completion_time(forecaster, df, use_story_points)
        elif choice == "3":
            forecast_backlog_completion(jira_manager, forecaster, df, use_story_points)
        elif choice == "4":
            logger.info("Exiting the program.")
            break
        else:
            logger.warning("Invalid choice. Please try again.")

def forecast_completed(forecaster: MonteCarloForecaster, df: pd.DataFrame, use_story_points: bool):
    unit = "story points" if use_story_points else "items"
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

def forecast_completion_time(forecaster: MonteCarloForecaster, df: pd.DataFrame, use_story_points: bool):
    unit = "story points" if use_story_points else "items"
    target = float(input(f"Enter the number of {unit} to complete: "))
    days_to_target = forecaster.run_simulation(df, target_count=target)
    print(f"\nEstimated days to complete {target} {unit}:")
    print(f"50% chance of completion within: {days_to_target[0]:.0f} days")
    print(f"75% chance of completion within: {days_to_target[1]:.0f} days")
    print(f"85% chance of completion within: {days_to_target[2]:.0f} days")
    print(f"95% chance of completion within: {days_to_target[3]:.0f} days")

def forecast_backlog_completion(jira_manager: JiraDataManager, forecaster: MonteCarloForecaster, df: pd.DataFrame, use_story_points: bool):
    unresolved_filter_id = prompt_for_filter(jira_manager, "Select the filter for unresolved items:")
    item_count, story_point_sum = jira_manager.get_unresolved_count(unresolved_filter_id)
    
    if use_story_points:
        unresolved_count = story_point_sum
        unit = "story points"
    else:
        unresolved_count = item_count
        unit = "items"
    
    logger.info(f"Found {item_count} unresolved items totaling {story_point_sum:.2f} story points in the selected filter.")
    logger.info(f"Using {unresolved_count:.2f} {unit} for forecasting based on selected mode.")
    
    inflation = float(input(f"Enter the number of additional {unit} expected (default 0): ") or 0)
    total = unresolved_count + inflation

    logger.info(f"Forecasting completion time for {total:.2f} {unit} (including {inflation} inflated {unit}):")

    days_to_target, completion_dates = forecaster.forecast_backlog(df, total)

    print(f"\nProjected completion dates for {total:.2f} {unit}:")
    print(f"50% chance of completion by: {completion_dates[0]} (in {days_to_target[0]:.0f} days)")
    print(f"75% chance of completion by: {completion_dates[1]} (in {days_to_target[1]:.0f} days)")
    print(f"85% chance of completion by: {completion_dates[2]} (in {days_to_target[2]:.0f} days)")
    print(f"95% chance of completion by: {completion_dates[3]} (in {days_to_target[3]:.0f} days)")

if __name__ == "__main__":
    main()
