
#!/usr/bin/env python

import pandas as pd
import pandas_gbq
import numpy as np
from datetime import datetime
import random

# Configuration
PROJECT_ID = "your-gcp-project-id"
BQ_VIEW = "your_dataset.ab_test_events"  # Must have: group (A/B), conversions (0 or 1)
LOG_PATH = "/home/yourname/logs/mab_test_log.txt"

# Fetch raw event data from BigQuery
def fetch_event_data():
    query = f'''
    SELECT group, user_id, converted
    FROM `{BQ_VIEW}`
    WHERE group IN ('A', 'B')
    '''
    return pandas_gbq.read_gbq(query, project_id=PROJECT_ID)

# Thompson Sampling implementation for 2 arms
def run_thompson_sampling(df):
    summary = df.groupby("group")["converted"].agg(["sum", "count"]).reset_index()
    summary.columns = ["group", "successes", "total"]

    # Initialize with pseudocounts (1 success, 1 failure) to avoid division by zero
    results = {}
    for _, row in summary.iterrows():
        group = row["group"]
        successes = row["successes"]
        failures = row["total"] - successes
        sample = np.random.beta(successes + 1, failures + 1)
        results[group] = {
            "successes": int(successes),
            "failures": int(failures),
            "sample": round(sample, 4)
        }

    best_group = max(results, key=lambda g: results[g]["sample"])
    results["winner"] = best_group
    return results

# Log the result
def log_results(results):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a") as f:
        f.write(f"[{timestamp}] Thompson Sampling MAB Result:\n")
        for group in ['A', 'B']:
            stats = results.get(group, {})
            f.write(f"  Group {group}: Successes={stats.get('successes', 0)}, Failures={stats.get('failures', 0)}, Sample={stats.get('sample', 0)}\n")
        f.write(f"  🎯 Winner: {results['winner']}\n\n")

# Main execution
def main():
    try:
        df = fetch_event_data()
        df["group"] = df["group"].str.strip().str.upper()  # Normalize group names
        df["converted"] = df["converted"].astype(int)
        results = run_thompson_sampling(df)
        log_results(results)
        print(f"✔️ MAB run complete. Winner: {results['winner']}")
    except Exception as e:
        with open(LOG_PATH, "a") as f:
            f.write(f"[{datetime.now()}] ERROR: {e}\n\n")
        print(f"❌ Error running MAB test: {e}")

if __name__ == "__main__":
    main()
