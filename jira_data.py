from requests.exceptions import RequestException
import concurrent.futures
from typing import List, Tuple, Optional, Dict
import logging
import pytz
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)

class JiraDataManager:
    def __init__(self, email: str, api_key: str, done_status: str = "Done"):
        self.email = email
        self.api_key = api_key
        self.auth = HTTPBasicAuth(email, api_key)
        self.base_url = "https://calendlyapp.atlassian.net"
        self.done_status = done_status

    def get_filter_list(self) -> List[Tuple[str, str]]:
        url = f"{self.base_url}/rest/api/3/filter/search"
        headers = {"Accept": "application/json"}
        params = {"accountId": self.get_account_id()}
        response = requests.get(url, headers=headers, auth=self.auth, params=params)
        if response.status_code == 200:
            filters = response.json()["values"]
            return [(f["id"], f["name"]) for f in filters]
        else:
            logger.error(f"Error fetching filters: {response.status_code}")
            return []

    def get_account_id(self) -> str:
        url = f"{self.base_url}/rest/api/3/myself"
        headers = {"Accept": "application/json"}
        response = requests.get(url, headers=headers, auth=self.auth)
        if response.status_code == 200:
            return response.json()["accountId"]
        else:
            logger.error(f"Error fetching account ID: {response.status_code}")
            return ""

    def get_ticket_data(self, filter_id: str) -> List[Dict]:
        url = f"{self.base_url}/rest/api/2/search"

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        all_issues = []
        start_at = 0
        max_results = 100  # Increased from 50 to 100 for efficiency

        while True:
            payload = {
                "jql": f"filter={filter_id}",
                "fields": ["key", "customfield_10026", "customfield_10082", "created", "resolutiondate", "assignee", "summary", "description", "status"],
                "expand": ["changelog"],
                "startAt": start_at,
                "maxResults": max_results
            }
            
            try:
                response = requests.post(url, json=payload, headers=headers, auth=self.auth)
                response.raise_for_status()  # Raises an HTTPError for bad responses
                
                data = response.json()
                issues = data["issues"]
                
                for issue in issues:
                    logger.debug(f"Received issue: Key: {issue['key']}, Story Points: {issue['fields'].get('customfield_10026')}")
                
                processed_issues = self.process_issues(issues)
                all_issues.extend(processed_issues)
                
                if len(issues) < max_results or len(all_issues) >= data["total"]:
                    break
                
                start_at += len(issues)
            
            except RequestException as e:
                logger.error(f"Error fetching tickets: {str(e)}")
                break

        logger.info(f"Retrieved {len(all_issues)} issues in total")
        return all_issues

    def process_issues(self, issues: List[Dict]) -> List[Dict]:
        processed_issues = []
        for issue in issues:
            cycle_times, backlog_to_progress = self.calculate_cycle_times(issue["changelog"]["histories"])
            processed_issue = {
                "key": issue["key"],
                "story_points": issue["fields"].get("customfield_10026"),
                "created_date": pd.to_datetime(issue["fields"]["created"]).date(),
                "completed_date": pd.to_datetime(issue["fields"]["resolutiondate"]).date() if issue["fields"]["resolutiondate"] else None,
                "assignee": issue["fields"]["assignee"]["displayName"] if issue["fields"]["assignee"] else None,
                "description_length": len(issue["fields"]["description"] or ""),
                "acceptance_criteria": issue["fields"].get("customfield_10082", ""),
                "acceptance_criteria_length": len(issue["fields"].get("customfield_10082", "") or ""),
                "cycle_times": cycle_times,
                "backlog_to_progress": backlog_to_progress
            }
            processed_issues.append(processed_issue)
        return processed_issues

    def calculate_cycle_times(self, changelog: List[Dict]) -> Tuple[Dict[str, timedelta], int]:
            status_changes = []
            current_status = None
            backlog_to_progress_count = 0

            for history in changelog:
                for item in history["items"]:
                    if item["field"] == "status":
                        timestamp = pd.to_datetime(history["created"]).tz_convert(pytz.UTC)
                        from_status = item["fromString"]
                        to_status = item["toString"]
                        status_changes.append((timestamp, from_status, to_status))

                        if from_status == "Backlog" and to_status == "In Progress":
                            backlog_to_progress_count += 1

            status_changes.sort(key=lambda x: x[0])

            cycle_times = {}
            status_start_times = {}

            for timestamp, from_status, to_status in status_changes:
                if from_status in status_start_times:
                    duration = timestamp - status_start_times[from_status]
                    if from_status in cycle_times:
                        cycle_times[from_status] += duration
                    else:
                        cycle_times[from_status] = duration
                    del status_start_times[from_status]

                status_start_times[to_status] = timestamp

            # Add time for the current status
            current_time = pd.Timestamp.now(tz=pytz.UTC)
            for status, start_time in status_start_times.items():
                duration = current_time - start_time
                if status in cycle_times:
                    cycle_times[status] += duration
                else:
                    cycle_times[status] = duration

            return cycle_times, backlog_to_progress_count

    def get_unresolved_count(self, filter_id: str) -> Tuple[int, float]:
        print("Here's the problem maybe.")
        url = f"{self.base_url}/rest/api/2/search"
        headers = {"Content-Type": "application/json"}
        params = {
            "jql": f"filter={filter_id} AND resolution IS EMPTY",
            "fields": "customfield_10026",
            "maxResults": 1000
        }

        item_count = 0
        story_point_sum = 0
        start_at = 0

        while True:
            params["startAt"] = start_at
            response = requests.get(url, headers=headers, params=params, auth=self.auth)

            if response.status_code != 200:
                logger.error(f"Error fetching unresolved tickets: {response.status_code}")
                return 0, 0

            data = response.json()
            issues = data["issues"]

            item_count += len(issues)
            story_point_sum += sum(issue["fields"].get("customfield_10026", 0) or 0 for issue in issues)

            if len(issues) < 1000:
                break

            start_at += len(issues)

        return item_count, story_point_sum

    def get_story_points(self, filter_id: str) -> List[Optional[float]]:
        url = f"{self.base_url}/rest/api/2/search"
        headers = {"Content-Type": "application/json"}
        params = {
            "jql": f"filter={filter_id} AND resolution IS EMPTY",
            "fields": "customfield_10026",
            "maxResults": 1000
        }
        
        story_points = []
        start_at = 0
        
        while True:
            params["startAt"] = start_at
            response = requests.get(url, headers=headers, params=params, auth=self.auth)
            
            if response.status_code != 200:
                logger.error(f"Error fetching story points: {response.status_code}")
                return []
            
            data = response.json()
            issues = data['issues']
            
            story_points.extend([issue['fields'].get('customfield_10026') for issue in issues])
            
            if len(issues) < 1000:
                break
            
            start_at += len(issues)
        
        return story_points



    def backfill_story_points(self, tickets: List[dict]) -> List[dict]:
        valid_points = [
            ticket["story_points"]
            for ticket in tickets
            if ticket["story_points"] is not None
        ]
        if not valid_points:
            logger.warning("No valid story points found. Using 1 as default.")
            average_points = 1
        else:
            average_points = np.mean(valid_points)

        logger.info(f"Average story points: {average_points:.2f}")

        backfilled_count = 0
        for ticket in tickets:
            if "story_points" not in ticket or ticket["story_points"] is None:
                ticket["story_points"] = average_points
                ticket["original_story_points"] = False
                backfilled_count += 1
            else:
                ticket["original_story_points"] = True

        logger.info(f"Backfilled {backfilled_count} tickets with average story points")

        return tickets

    def prepare_data(self, tickets: List[Dict], use_story_points: bool = False) -> pd.DataFrame:
        logger.info(f"Processing {len(tickets)} tickets...")

        closed_tickets = [t for t in tickets if t['completed_date'] is not None]

        if not closed_tickets:
            logger.warning("No closed tickets found.")
            return None

        earliest_date = min(t['completed_date'] for t in closed_tickets)
        latest_date = max(t['completed_date'] for t in closed_tickets)

        logger.info(f"Data spans from {earliest_date} to {latest_date}")
        logger.info(f"Total completed tickets in this period: {len(closed_tickets)}")

        date_range = pd.date_range(start=earliest_date, end=latest_date, freq="D")
        df = pd.DataFrame(index=date_range, columns=["Completed"])
        df.index.name = "Date"
        df["Completed"] = 0  # Initialize all days with 0 completed items/points

        for ticket in closed_tickets:
            closed_timestamp = pd.Timestamp(ticket['completed_date'])
            if closed_timestamp in df.index:
                value = ticket['story_points'] if use_story_points else 1
                df.at[closed_timestamp, 'Completed'] += value
            else:
                logger.warning(f"Date {closed_timestamp} not in date range.")
                
        df["Cumulative Completed"] = df["Completed"].cumsum()

        average_completion_rate = df["Completed"].mean()
        unit = "story points" if use_story_points else "items"
        logger.info(f"Average completion rate: {average_completion_rate:.2f} {unit} per day")

        return df

    def get_correlations(self, tickets: List[Dict]) -> Dict:
        df = pd.DataFrame(tickets)
        
        correlations = {}
        
        def timedelta_to_seconds(td):
            return td.total_seconds() if pd.notnull(td) else pd.NaT

        if 'acceptance_criteria_length' in df.columns and 'cycle_time' in df.columns:
            cycle_times = df['cycle_time'].apply(timedelta_to_seconds)
            correlations['acceptance_criteria_cycle_time'] = df['acceptance_criteria_length'].corr(cycle_times)

        if 'description_length' in df.columns and 'cycle_time' in df.columns:
            cycle_times = df['cycle_time'].apply(timedelta_to_seconds)
            correlations['description_cycle_time'] = df['description_length'].corr(cycle_times)

        if 'story_points' in df.columns and 'cycle_time' in df.columns:
            df_with_points = df.dropna(subset=['story_points', 'cycle_time'])
            cycle_times = df_with_points['cycle_time'].apply(timedelta_to_seconds)
            correlations['story_points_cycle_time'] = df_with_points['story_points'].corr(cycle_times)

        return correlations
