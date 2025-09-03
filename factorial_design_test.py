#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 10:25:22 2025

@author: maxvargas
"""

# define factors
# button color: blue or green x screen location: top or bottom
# 2x2 with 4 total groups: blue/top, blue/bottom, green/top, green/bottom

# import pckgs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

# define experiment
button_colors = ['blue','green']
screen_locations = ['top','bottom']
n_per_group = 100 # users per group
np.random.seed(42)

# generate synthetic data
data = []
for color in button_colors:
    for location in screen_locations:
        for _ in range(n_per_group):
            base = 0.10
            if color == 'green':
                base += 0.03
            if location == 'bottom':
                base += 0.02
            if color == 'green' and location == 'bottom':
                base += 0.01  # interaction
            converted = np.random.binomial(1, base)
            data.append([color, location, converted])

df = pd.DataFrame(data, columns=['button_color', 'screen_location', 'converted'])

# run factorial ANOVA
model = ols('converted ~ C(button_color) * C(screen_location)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("\nANOVA Results:")
print(anova_table)

# visualize results
grouped = df.groupby(['button_color', 'screen_location'])['converted'].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(data=grouped, x='button_color', y='converted', hue='screen_location')
plt.title('Conversion Rate by Button Color and Screen Location')
plt.ylabel('Conversion Rate')
plt.ylim(0, 0.25)
plt.grid(True)
plt.tight_layout()
plt.show()

# Putting the green button at the bottom doesn’t give an extra conversion boost 
# beyond what each factor contributes on its own — and none of those are statistically significant either