#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 09:25:05 2025

@author: maxvargas
"""

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter

# simulate data
np.random.seed(42)
n = 2000
data = pd.DataFrame({
    "user_id": [f"user_{i}" for i in range(n)],
    "group": np.random.choice(["A","B"], size=n),
    "is_premium_user": np.random.choice([0,1], size=n, p=[0.7,0.3]),
    "age": np.random.normal(loc=35, scale=10, size=n).astype(int)
})

base = np.random.exponential(scale=10, size=n)
group_effect = np.where(data["group"] == "B", np.random.exponential(scale=5, size=n), 0)
premium_effect = np.where(data["is_premium_user"] == 1, np.random.exponential(scale=7, size=n), 0)
data["duration"] = (base + group_effect + premium_effect).round(1)
data["event_observed"] = np.random.binomial(1, 0.8, size=n)

# Kaplan-Meier survival curve
kmf = KaplanMeierFitter()
plt.figure(figsize=(10, 6))
for label, df_group in data.groupby("group"):
    kmf.fit(df_group["duration"], df_group["event_observed"], label=f"Group {label}")
    kmf.plot_survival_function(ci_show=False)
plt.title("Survival Curves by Group")
plt.xlabel("Days")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.tight_layout()
plt.show()

# Cox Proportional Hazards Model
cph_data = data[["duration", "event_observed", "is_premium_user", "age"]].copy()
cph_data["group_B"] = (data["group"] == "B").astype(int)
cph = CoxPHFitter()
cph.fit(cph_data, duration_col="duration", event_col="event_observed")
cph.print_summary()
cph.plot()
plt.title("CoxPH Feature Effects")
plt.grid(True)
plt.tight_layout()
plt.show()

# The KM curve chart shows the percentage of users who are still active over time, 
# starting from the day they signed up. Each line represents a different group. 
# Group B consistently retains more users over time than Group A.

# Being a premium user reduces the risk of churn by 39%. 
# This means premium users stay active longer, even after adjusting for age and test group. 
# This reinforces the value of upselling free users
# Users in Group B are 33% less likely to churn than those in Group A. 
# This suggests that whatever feature or design we tested in Group B is positively influencing retention.
# While older users show slightly lower churn, the effect isn’t statistically significant in this model.