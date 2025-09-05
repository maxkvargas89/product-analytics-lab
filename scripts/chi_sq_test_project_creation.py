#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 09:47:48 2025

@author: maxvargas
"""

import pandas as pd
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration
file_path = "/Users/maxvargas/code_testing/data/first_competency_project_creation.csv"
df = pd.read_csv(file_path)

contingency = pd.crosstab(df['first_competency_clicked'], df['project_created'])

print("Contingency Table:")
print(contingency)

# Run the chi-squared test
chi2, p, dof, expected = chi2_contingency(contingency)

print(f"\nChi-Squared Statistic: {chi2:.4f}")
print(f"Degrees of Freedom: {dof}")
print(f"P-Value: {p:.4f}")

# Expected counts under the null
expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)
print("\nExpected Counts:")
print(expected_df)

# Results interpretation
alpha = 0.05

if p < alpha:
    print("\n✅ There is a statistically significant relationship between first competency clicked and project creation.")
else:
    print("\n❌ There is no significant relationship between first competency clicked and project creation.")
    
# Bar plot of project creation rate by first competency clicked
conversion_rate = df.groupby('first_competency_clicked')['project_created'].mean().reset_index()

plt.figure(figsize=(8,5))
sns.barplot(data=conversion_rate, x='first_competency_clicked', y='project_created', palette='Set2')
plt.title('Project Creation Rate by First Competency Clicked')
plt.ylabel('Conversion Rate')
plt.xlabel('First Competency Clicked')
plt.ylim(0, 0.2)
plt.grid(True, axis='y', linestyle='--')
plt.tight_layout()
plt.show()

# We tested whether the first learning competency a user clicks on is associated with their likelihood of creating a project. 
# The analysis shows a statistically significant difference (p = 0.0064). 
# This means the type of content users first engage with likely influences their commitment or activation.
# Specifically, users who began with Creativity or Communication showed higher-than-expected project creation.