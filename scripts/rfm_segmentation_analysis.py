#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 15:40:34 2025

@author: maxvargas
"""

# Simulate Raw User Activity Data
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
n_users = 1000
today = datetime(2025, 9, 4)

users = pd.DataFrame({
    'user_id': np.arange(1, n_users + 1),
    'last_visit_date': [today - timedelta(days=np.random.poisson(15)) for _ in range(n_users)],
    'clicks_on_competencies': np.random.poisson(lam=12, size=n_users),
    'resource_engagements': np.random.poisson(lam=5, size=n_users),
    'created_project': np.random.binomial(1, p=0.15, size=n_users)
})

# Create RFM features
users['recency'] = (today - users['last_visit_date']).dt.days
users['frequency'] = users['clicks_on_competencies']
users['engagement'] = users['resource_engagements']

# Score R, F, and M Using Quintiles
users['r_score'] = pd.qcut(users['recency'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
users['f_score'] = pd.qcut(users['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
users['m_score'] = pd.qcut(users['engagement'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)

# Combine into FRM Segment and Score
users['rfm_segment'] = users['r_score'].astype(str) + users['f_score'].astype(str) + users['m_score'].astype(str)
users['rfm_score'] = users[['r_score', 'f_score', 'm_score']].sum(axis=1)

# Label Segments Based on RFM Score
def rfm_label(row):
    if row['rfm_score'] >= 13:
        return 'Champions'
    elif row['rfm_score'] >= 10:
        return 'Loyal'
    elif row['rfm_score'] >= 7:
        return 'Potential'
    elif row['rfm_score'] >= 4:
        return 'Needs Attention'
    else:
        return 'At Risk'

users['segment'] = users.apply(rfm_label, axis=1)

# Analyze Conversion Rates by Segment
conversion_by_segment = users.groupby('segment')['created_project'].agg(['count', 'sum', 'mean']).reset_index()
conversion_by_segment.columns = ['segment', 'users', 'converted', 'conversion_rate']
print(conversion_by_segment.sort_values('conversion_rate', ascending=False))

# Visualize the Results
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
sns.barplot(data=conversion_by_segment.sort_values('conversion_rate', ascending=False),
            x='conversion_rate', y='segment', palette='viridis')
plt.xlabel('Project Creation Conversion Rate')
plt.ylabel('RFM Segment')
plt.title('Conversion Rate by RFM Segment')
plt.grid(True)
plt.tight_layout()
plt.show()

#We segmented users based on their recency, frequency, and depth of engagement with our learning competencies. We found clear differences in project creation rates across these segments, which can inform how we prioritize outreach and feature nudges.
