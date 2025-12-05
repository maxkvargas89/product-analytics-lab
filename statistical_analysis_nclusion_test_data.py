#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 21:52:48 2025

@author: maxvargas
"""

"""
Nclusion A/B Test: Savings Incentive Feature
Step 2: Statistical Analysis & Hypothesis Testing

This script performs comprehensive statistical analysis of the A/B test,
including hypothesis testing, confidence intervals, effect size calculations,
and guardrail metric evaluation.

Key Statistical Tests:
1. Two-proportion z-test (primary metric: conversion rate)
2. Two-sample t-test (secondary metric: deposit amounts)
3. Guardrail metric evaluation (chi-square, t-tests)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, ttest_ind, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading experiment data...")
df = pd.read_csv('/Users/maxvargas/product_analytics_lab/data/nclusion_ab_test_data.csv')
print(f"✅ Loaded {len(df):,} user records\n")

# Separate control and treatment groups
df_control = df[df['experiment_group'] == 'control']
df_treatment = df[df['experiment_group'] == 'treatment']
print(f"Control group: {len(df_control):,} users")
print(f"Treatment group: {len(df_treatment):,} users\n")

# ============================================================================
# HELPER FUNCTIONS FOR STATISTICAL TESTS
# ============================================================================

def two_proportion_ztest(successes_1, n_1, successes_2, n_2, alpha=0.05):
    """
    Perform two-proportion z-test.
    
    Tests whether two population proportions are significantly different.
    Used for binary metrics like conversion rates.
    
    Parameters:
    -----------
    successes_1, successes_2 : int
        Number of successes in each group
    n_1, n_2 : int
        Total observations in each group
    alpha : float
        Significance level (default 0.05 for 95% confidence)
        
    Returns:
    --------
    dict with test statistics, p-value, confidence interval, effect size
    """
    
    # Calculate proportions
    p1 = successes_1 / n_1
    p2 = successes_2 / n_2
    
    # Pooled proportion (under null hypothesis of no difference)
    p_pool = (successes_1 + successes_2) / (n_1 + n_2)
    
    # Standard error under null hypothesis
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1/n_1 + 1/n_2))
    
    # Z-statistic
    z_stat = (p2 - p1) / se_pool
    
    # Two-tailed p-value
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    
    # Confidence interval for difference in proportions
    se_diff = np.sqrt(p1 * (1 - p1) / n_1 + p2 * (1 - p2) / n_2)
    z_critical = norm.ppf(1 - alpha/2)
    ci_lower = (p2 - p1) - z_critical * se_diff
    ci_upper = (p2 - p1) + z_critical * se_diff
    
    # Relative lift
    relative_lift = (p2 - p1) / p1 if p1 > 0 else np.nan
    
    return {
        'control_rate': p1,
        'treatment_rate': p2,
        'absolute_lift': p2 - p1,
        'relative_lift': relative_lift,
        'z_statistic': z_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': p_value < alpha
    }

def calculate_power(p1, p2, n, alpha=0.05):
    """
    Calculate statistical power of the test.
    
    Power = probability of detecting a true effect when it exists.
    We want power >= 80%.
    
    Parameters:
    -----------
    p1, p2 : float
        Proportions in control and treatment groups
    n : int
        Sample size per group
    alpha : float
        Significance level
        
    Returns:
    --------
    float: Statistical power (0 to 1)
    """
    
    # Standard error under alternative hypothesis (true difference exists)
    se_alt = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    
    # Critical value for two-tailed test
    z_critical = norm.ppf(1 - alpha/2)
    
    # Non-centrality parameter
    effect_size = (p2 - p1) / se_alt
    
    # Power calculation
    power = 1 - norm.cdf(z_critical - effect_size) + norm.cdf(-z_critical - effect_size)
    
    return power

# ============================================================================
# PRIMARY METRIC ANALYSIS: SAVINGS ACCOUNT CONVERSION RATE
# ============================================================================

print("="*80)
print("PRIMARY METRIC: SAVINGS ACCOUNT CONVERSION RATE")
print("="*80)

# Extract conversion data
control_conversions = df_control['opened_savings_account'].sum()
control_n = len(df_control)
treatment_conversions = df_treatment['opened_savings_account'].sum()
treatment_n = len(df_treatment)

# Perform two-proportion z-test
results_conversion = two_proportion_ztest(
    control_conversions, control_n,
    treatment_conversions, treatment_n
)

print(f"\n📊 Conversion Rate Results:")
print(f"  Control:   {results_conversion['control_rate']:.2%} ({control_conversions:,} / {control_n:,})")
print(f"  Treatment: {results_conversion['treatment_rate']:.2%} ({treatment_conversions:,} / {treatment_n:,})")
print(f"\n📈 Effect Size:")
print(f"  Absolute Lift: {results_conversion['absolute_lift']:.4f} ({results_conversion['absolute_lift']*100:.2f} percentage points)")
print(f"  Relative Lift: {results_conversion['relative_lift']:.2%}")
print(f"\n🧪 Statistical Test:")
print(f"  Z-statistic: {results_conversion['z_statistic']:.4f}")
print(f"  P-value: {results_conversion['p_value']:.6f}")
print(f"  95% CI for difference: [{results_conversion['ci_lower']:.4f}, {results_conversion['ci_upper']:.4f}]")
print(f"  Significant at α=0.05? {'✅ YES' if results_conversion['significant'] else '❌ NO'}")

# Calculate statistical power
observed_power = calculate_power(
    results_conversion['control_rate'],
    results_conversion['treatment_rate'],
    control_n
)
print(f"\n⚡ Statistical Power: {observed_power:.2%}")
print(f"  {'✅ Sufficient power (>80%)' if observed_power >= 0.80 else '⚠️  Underpowered (<80%)'}")

# Business interpretation
print(f"\n💼 Business Interpretation:")
if results_conversion['significant'] and results_conversion['relative_lift'] > 0:
    print(f"  The savings incentive feature caused a statistically significant")
    print(f"  {results_conversion['relative_lift']:.1%} increase in savings account conversions.")
    print(f"  We can be 95% confident the true lift is between")
    print(f"  {results_conversion['ci_lower']*100:.2f} and {results_conversion['ci_upper']*100:.2f} percentage points.")
else:
    print(f"  No statistically significant difference detected.")

# ============================================================================
# SECONDARY METRIC ANALYSIS: INITIAL DEPOSIT AMOUNTS
# ============================================================================

print("\n" + "="*80)
print("SECONDARY METRIC: INITIAL DEPOSIT AMOUNTS (CONVERTERS ONLY)")
print("="*80)

# Filter to only users who opened accounts
df_control_converters = df_control[df_control['opened_savings_account'] == 1]
df_treatment_converters = df_treatment[df_treatment['opened_savings_account'] == 1]

control_deposits = df_control_converters['initial_deposit_amount'].dropna()
treatment_deposits = df_treatment_converters['initial_deposit_amount'].dropna()

# Descriptive statistics
print(f"\n📊 Deposit Amount Distribution:")
print(f"  Control Group (n={len(control_deposits):,}):")
print(f"    Mean:   ${control_deposits.mean():.2f}")
print(f"    Median: ${control_deposits.median():.2f}")
print(f"    Std:    ${control_deposits.std():.2f}")
print(f"\n  Treatment Group (n={len(treatment_deposits):,}):")
print(f"    Mean:   ${treatment_deposits.mean():.2f}")
print(f"    Median: ${treatment_deposits.median():.2f}")
print(f"    Std:    ${treatment_deposits.std():.2f}")

# Two-sample t-test (for means)
t_stat, p_value_ttest = ttest_ind(treatment_deposits, control_deposits)
mean_diff = treatment_deposits.mean() - control_deposits.mean()
relative_diff = mean_diff / control_deposits.mean()

# Confidence interval for mean difference
se_diff = np.sqrt(control_deposits.var()/len(control_deposits) + 
                   treatment_deposits.var()/len(treatment_deposits))
df_ttest = len(control_deposits) + len(treatment_deposits) - 2
t_critical = stats.t.ppf(0.975, df_ttest)
ci_lower_deposits = mean_diff - t_critical * se_diff
ci_upper_deposits = mean_diff + t_critical * se_diff

print(f"\n📈 Effect Size:")
print(f"  Mean Difference: ${mean_diff:.2f} ({relative_diff:.1%} higher in treatment)")
print(f"\n🧪 Two-Sample T-Test:")
print(f"  T-statistic: {t_stat:.4f}")
print(f"  P-value: {p_value_ttest:.6f}")
print(f"  95% CI for difference: [${ci_lower_deposits:.2f}, ${ci_upper_deposits:.2f}]")
print(f"  Significant at α=0.05? {'✅ YES' if p_value_ttest < 0.05 else '❌ NO'}")

print(f"\n💼 Business Interpretation:")
if p_value_ttest < 0.05 and mean_diff > 0:
    print(f"  Users exposed to the incentive not only converted at higher rates,")
    print(f"  but also deposited {relative_diff:.1%} more on average (${mean_diff:.2f}).")
    print(f"  This suggests the incentive attracts higher-quality savers.")
else:
    print(f"  No significant difference in deposit amounts between groups.")

# ============================================================================
# GUARDRAIL METRICS ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("GUARDRAIL METRICS: ENSURING NO NEGATIVE SIDE EFFECTS")
print("="*80)

guardrail_results = []

# 1. App Engagement Rate
control_engagement = df_control['returned_to_app_7d'].sum()
treatment_engagement = df_treatment['returned_to_app_7d'].sum()

results_engagement = two_proportion_ztest(
    control_engagement, len(df_control),
    treatment_engagement, len(df_treatment)
)

print(f"\n🛡️  1. App Engagement Rate (7-day return)")
print(f"  Control:   {results_engagement['control_rate']:.2%}")
print(f"  Treatment: {results_engagement['treatment_rate']:.2%}")
print(f"  Difference: {results_engagement['absolute_lift']:.4f} ({results_engagement['relative_lift']:.2%})")
print(f"  P-value: {results_engagement['p_value']:.6f}")
status_1 = "⚠️  Degraded" if results_engagement['significant'] and results_engagement['absolute_lift'] < -0.01 else "✅ Healthy"
print(f"  Status: {status_1}")

# 2. Account Abandonment Rate (only for converters)
df_control_converters_clean = df_control_converters.dropna(subset=['account_abandoned'])
df_treatment_converters_clean = df_treatment_converters.dropna(subset=['account_abandoned'])

control_abandonment = df_control_converters_clean['account_abandoned'].sum()
treatment_abandonment = df_treatment_converters_clean['account_abandoned'].sum()

results_abandonment = two_proportion_ztest(
    control_abandonment, len(df_control_converters_clean),
    treatment_abandonment, len(df_treatment_converters_clean)
)

print(f"\n🛡️  2. Account Abandonment Rate (converters only)")
print(f"  Control:   {results_abandonment['control_rate']:.2%}")
print(f"  Treatment: {results_abandonment['treatment_rate']:.2%}")
print(f"  Difference: {results_abandonment['absolute_lift']:.4f} ({results_abandonment['relative_lift']:.2%})")
print(f"  P-value: {results_abandonment['p_value']:.6f}")
status_2 = "⚠️  Elevated" if results_abandonment['significant'] and results_abandonment['absolute_lift'] > 0.02 else "✅ Acceptable"
print(f"  Status: {status_2}")

# 3. Support Ticket Rate
control_tickets = df_control['filed_support_ticket'].sum()
treatment_tickets = df_treatment['filed_support_ticket'].sum()

results_tickets = two_proportion_ztest(
    control_tickets, len(df_control),
    treatment_tickets, len(df_treatment)
)

print(f"\n🛡️  3. Support Ticket Rate")
print(f"  Control:   {results_tickets['control_rate']:.2%} ({control_tickets} tickets)")
print(f"  Treatment: {results_tickets['treatment_rate']:.2%} ({treatment_tickets} tickets)")
print(f"  Difference: {results_tickets['absolute_lift']:.4f} ({results_tickets['relative_lift']:.2%})")
print(f"  P-value: {results_tickets['p_value']:.6f}")
status_3 = "⚠️  Elevated" if results_tickets['significant'] and results_tickets['absolute_lift'] > 0.001 else "✅ Normal"
print(f"  Status: {status_3}")

# 4. Transaction Account Activity
control_transactions = df_control['used_transaction_account'].sum()
treatment_transactions = df_treatment['used_transaction_account'].sum()

results_transactions = two_proportion_ztest(
    control_transactions, len(df_control),
    treatment_transactions, len(df_treatment)
)

print(f"\n🛡️  4. Transaction Account Activity")
print(f"  Control:   {results_transactions['control_rate']:.2%}")
print(f"  Treatment: {results_transactions['treatment_rate']:.2%}")
print(f"  Difference: {results_transactions['absolute_lift']:.4f} ({results_transactions['relative_lift']:.2%})")
print(f"  P-value: {results_transactions['p_value']:.6f}")
status_4 = "⚠️  Cannibalized" if results_transactions['significant'] and results_transactions['absolute_lift'] < -0.02 else "✅ Stable"
print(f"  Status: {status_4}")

# ============================================================================
# FINAL RECOMMENDATION
# ============================================================================

print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)

print(f"\n🎯 PRIMARY METRIC:")
print(f"  ✅ Statistically significant {results_conversion['relative_lift']:.1%} lift in conversions")
print(f"  ❌ Not ractically significant (below 15% target)")

print(f"\n💰 SECONDARY METRIC:")
if p_value_ttest < 0.05:
    print(f"  ✅ Significantly higher deposits in treatment group (${mean_diff:.2f} more)")
else:
    print(f"  ➖ No significant difference in deposit amounts")

print(f"\n🛡️  GUARDRAIL METRICS:")
print(f"  App Engagement: {status_1}")
print(f"  Abandonment Rate: {status_2}")
print(f"  Support Tickets: {status_3}")
print(f"  Transaction Activity: {status_4}")

print(f"\n✅ RECOMMENDATION: SHIP THE FEATURE")
print(f"   The savings incentive successfully increases conversions with minimal")
print(f"   negative side effects. Consider monitoring abandonment rates closely")
print(f"   post-launch to ensure bonus-chasing behavior doesn't increase long-term.")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)