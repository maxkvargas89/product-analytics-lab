
#!/usr/bin/env python

import pandas as pd
import pandas_gbq
import numpy as np
from datetime import datetime
import random

# Configuration
PROJECT_ID = "your-gcp-project-id"
BQ_VIEW = "your_dataset.ab_test_events"  # Must have: group (A/B), conversions (0 or 1)
LOG_PATH = "/home/yourname/logs/mab_egreedy_log.txt"
EPSILON = 0.1  # 10% of the time we explore

# Fetch raw event data from BigQuery
def fetch_event_data():
    query = f'''
    SELECT group, user_id, converted
    FROM `{BQ_VIEW}`
    WHERE group IN ('A', 'B')
    '''
    return pandas_gbq.read_gbq(query, project_id=PROJECT_ID)

# ε-Greedy algorithm implementation
def run_egreedy(df, epsilon=EPSILON):
    summary = df.groupby("group")["converted"].agg(["sum", "count"]).reset_index()
    summary.columns = ["group", "successes", "total"]
    summary["conversion_rate"] = summary["successes"] / summary["total"]

    group_a = summary[summary["group"] == "A"]
    group_b = summary[summary["group"] == "B"]

    if random.random() < epsilon:
        winner = random.choice(["A", "B"])
        explored = True
    else:
        winner = "A" if group_a["conversion_rate"].values[0] > group_b["conversion_rate"].values[0] else "B"
        explored = False

    results = {
        "conversion_rate_A": round(group_a["conversion_rate"].values[0], 4),
        "conversion_rate_B": round(group_b["conversion_rate"].values[0], 4),
        "explored": explored,
        "winner": winner
    }
    return results

# Log the result
def log_results(results):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a") as f:
        f.write(f"[{timestamp}] ε-Greedy MAB Result:\n")
        f.write(f"  Conversion Rate A: {results['conversion_rate_A']}\n")
        f.write(f"  Conversion Rate B: {results['conversion_rate_B']}\n")
        f.write(f"  Winner: {results['winner']}\n")
        f.write(f"  Exploration? {'Yes' if results['explored'] else 'No'}\n\n")

# Main execution
def main():
    try:
        df = fetch_event_data()
        df["group"] = df["group"].str.strip().str.upper()  # Normalize group names
        df["converted"] = df["converted"].astype(int)
        results = run_egreedy(df)
        log_results(results)
        print(f"✔️ ε-Greedy MAB run complete. Winner: {results['winner']} (Exploration: {results['explored']})")
    except Exception as e:
        with open(LOG_PATH, "a") as f:
            f.write(f"[{datetime.now()}] ERROR: {e}\n\n")
        print(f"❌ Error running ε-Greedy MAB test: {e}")

if __name__ == "__main__":
    main()
