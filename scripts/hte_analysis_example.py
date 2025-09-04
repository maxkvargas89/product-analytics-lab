
# Heterogeneous Treatment Effect (HTE) Analysis Example
# -----------------------------------------------------
# Simulates an A/B test with a hidden effect in mobile users only
# Includes: data generation, ANOVA with interaction, visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Set seed for reproducibility
np.random.seed(42)

# Step 1: Simulate experimental data
n = 1000
df = pd.DataFrame({
    'user_id': range(n),
    'treatment': np.random.choice([0, 1], size=n),  # 0 = control, 1 = treatment
    'platform': np.random.choice(['mobile', 'desktop'], size=n)  # Segment
})

# Step 2: Simulate conversion probabilities
df['base_rate'] = 0.08  # 8% base conversion
df['true_effect'] = np.where((df['treatment'] == 1) & (df['platform'] == 'mobile'), 0.06, 0.00)
df['prob_conversion'] = df['base_rate'] + df['true_effect']
df['converted'] = np.random.binomial(1, df['prob_conversion'])

# Step 3: Summary stats
overall = df.groupby('treatment')['converted'].mean().reset_index()
print("Overall conversion rates by treatment:")
print(overall)

# Step 4: Conversion rate by platform + treatment
grouped = df.groupby(['platform', 'treatment'])['converted'].mean().reset_index()
print("\nConversion rates by platform and treatment:")
print(grouped)

# Step 5: ANOVA model with interaction
model = ols("converted ~ C(treatment) * C(platform)", data=df).fit()
anova_results = sm.stats.anova_lm(model, typ=2)
print("\nANOVA Results:")
print(anova_results)

# Step 6: Plot the results
plt.figure(figsize=(8, 5))
sns.barplot(data=grouped, x='platform', y='converted', hue='treatment')
plt.title('Conversion Rate by Platform and Treatment')
plt.ylabel('Conversion Rate')
plt.xlabel('Platform')
plt.ylim(0, 0.20)
plt.legend(title='Treatment', labels=['Control', 'Treatment'])
plt.grid(True)
plt.tight_layout()
plt.show()
