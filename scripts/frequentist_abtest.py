#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 10:48:58 2025

@author: maxvargas
"""

# Imports

import pandas as pd
import scipy.stats as stats
from datetime import datetime

# CONFIGURATION
file_path = "/Users/maxvargas/code_testing/simulated_ab_test_data.csv"
ab_test_file = pd.read_csv(file_path)
ab_test_file_ready = ab_test_file.groupby('group').agg(
    conversions=('conversion_at','count'),
    total_users=('user_id','count'))
ab_test_file_ready = ab_test_file_ready.reset_index()

LOG_PATH = "/Users/maxvargas/code_testing/ab_test_log.txt"
ALPHA = 0.05

# Function to perform z-test for proportions
def perform_z_test(df):
    conv_a = df[df['group'] == 'A']['conversions'].values[0]
    users_a = df[df['group'] == 'A']['total_users'].values[0]
    conv_b = df[df['group'] == 'B']['conversions'].values[0]
    users_b = df[df['group'] == 'B']['total_users'].values[0]

    p1 = conv_a / users_a
    p2 = conv_b / users_b
    p_pool = (conv_a + conv_b) / (users_a + users_b)
    se = ((p_pool * (1 - p_pool)) * (1/users_a + 1/users_b)) ** 0.5
    z_score = (p2 - p1) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    return {
        "conversion_rate_A": round(p1, 4),
        "conversion_rate_B": round(p2, 4),
        "z_score": round(z_score, 4),
        "p_value": round(p_value, 4),
        "statistically_significant": p_value < ALPHA
    }

# Function to log results
def log_results(results):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a") as f:
        f.write(f"[{timestamp}] A/B Test Result:\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

# Main execution block
def main():
    try:
        df = ab_test_file_ready
        results = perform_z_test(df)
        log_results(results)
        print("A/B test results logged successfully.")
    except Exception as e:
        with open(LOG_PATH, "a") as f:
            f.write(f"[{datetime.now()}] ERROR: {e}\n\n")
        print(f"Error during A/B test: {e}")

if __name__ == "__main__":
    main()
