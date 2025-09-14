#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 15:28:05 2025

@author: maxvargas
"""

"""
The Lingo mobile app is experimenting with its onboarding process. 
The goal is to improve user engagement in the first 7 days of usage.

The product team is testing three different onboarding flows (A, B, and C) and two notification styles (calm vs. urgent). These are being run as a multivariate test (3x2 = 6 total variants).

Your task is to analyze the impact of these combinations on 7-day retention.

You have the following mock tables:

users: includes onboarding flow and notification style assignment
app_events: includes user activity and event timestamps
"""


# Set up mock data
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

np.random.seed(42)

# Define 6 experimental groups (3 onboarding x 2 notifications)
onboarding_flows = ['A', 'B', 'C']
notification_styles = ['calm', 'urgent']

# Create users
users = pd.DataFrame({
    'user_id': range(1, 601),
    'onboarding_flow': np.random.choice(onboarding_flows, size=600),
    'notification_style': np.random.choice(notification_styles, size=600),
    'signup_date': pd.to_datetime('2025-07-01')
})

# Simulate retention (1 if user opened app on day 7)
def simulate_retention(row):
    base_rate = 0.45
    modifiers = {
        ('A', 'calm'): 0.00,
        ('A', 'urgent'): 0.03,
        ('B', 'calm'): 0.02,
        ('B', 'urgent'): 0.04,
        ('C', 'calm'): -0.01,
        ('C', 'urgent'): 0.01
    }
    p = base_rate + modifiers[(row['onboarding_flow'], row['notification_style'])]
    return np.random.rand() < p

users['retained_day_7'] = users.apply(simulate_retention, axis=1).astype(int)

# Analyze retention by variant
retention_summary = users.groupby(['onboarding_flow', 'notification_style'])['retained_day_7'].mean().unstack()
print(retention_summary)

# Plot heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(retention_summary, annot=True, cmap="YlGnBu", fmt=".2%")
plt.title("7-Day Retention by Onboarding Flow and Notification Style")
plt.ylabel("Onboarding Flow")
plt.xlabel("Notification Style")
plt.show()

# Two-way anova
model = ols('retained_day_7 ~ C(onboarding_flow) * C(notification_style)', data=users).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)




