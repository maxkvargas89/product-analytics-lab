#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 21:44:54 2025

@author: maxvargas
"""

"""
Nclusion A/B Test: Savings Incentive Feature
Step 1: Generate Synthetic Experiment Data

This script creates realistic synthetic data for an A/B test evaluating
a temporary savings incentive feature on Nclusion's platform.

Business Context:
- Control group sees standard savings messaging
- Treatment group sees incentive messaging (2% bonus interest for 90 days)
- Goal: Increase savings account adoption and initial deposit amounts
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# EXPERIMENT PARAMETERS
# ============================================================================

# Sample sizes
N_CONTROL = 12000
N_TREATMENT = 12000
N_TOTAL = N_CONTROL + N_TREATMENT

# Baseline conversion rates (control group)
BASELINE_CONVERSION_RATE = 0.08  # 8% open savings accounts

# Treatment effect (15% relative lift = 0.08 * 1.15 = 0.092 = 9.2%)
TREATMENT_LIFT = 0.15
TREATMENT_CONVERSION_RATE = BASELINE_CONVERSION_RATE * (1 + TREATMENT_LIFT)

# Initial deposit amounts (for users who convert)
# Control group: lower deposits on average
CONTROL_DEPOSIT_MEAN = 150
CONTROL_DEPOSIT_STD = 100

# Treatment group: slightly higher deposits (incentive attracts more serious savers)
TREATMENT_DEPOSIT_MEAN = 175
TREATMENT_DEPOSIT_STD = 110

# Guardrail metric parameters
BASELINE_APP_ENGAGEMENT = 0.65  # 65% return to app within 7 days
BASELINE_ABANDONMENT_RATE = 0.12  # 12% open account but never deposit again
BASELINE_SUPPORT_TICKETS_PER_1000 = 8  # 8 tickets per 1000 users
BASELINE_TRANSACTION_ACTIVITY = 0.78  # 78% continue using checking accounts

# ============================================================================
# GENERATE USER DATA
# ============================================================================

def generate_user_cohort(n_users, group_name, conversion_rate, deposit_mean, deposit_std):
    """
    Generate synthetic user data for a single experimental group.
    
    Parameters:
    -----------
    n_users : int
        Number of users in this group
    group_name : str
        'control' or 'treatment'
    conversion_rate : float
        Probability of opening a savings account
    deposit_mean : float
        Mean initial deposit amount (for converters)
    deposit_std : float
        Standard deviation of initial deposit amounts
        
    Returns:
    --------
    pd.DataFrame with user-level experiment data
    """
    
    # Generate user IDs
    if group_name == 'control':
        user_ids = [f'user_{i:06d}' for i in range(1, n_users + 1)]
    else:
        user_ids = [f'user_{i:06d}' for i in range(n_users + 1, 2 * n_users + 1)]
    
    # Assign experiment group
    experiment_group = [group_name] * n_users
    
    # Generate exposure timestamp (random times over 30-day period)
    start_date = datetime(2025, 12, 4)
    exposure_dates = [start_date + timedelta(days=np.random.randint(0, 30)) 
                      for _ in range(n_users)]
    
    # PRIMARY METRIC: Did user open a savings account?
    opened_savings_account = np.random.binomial(1, conversion_rate, n_users)
    
    # SECONDARY METRIC: Initial deposit amount (only for converters)
    # Use log-normal distribution to create realistic right-skewed deposit amounts
    # Ensure minimum deposit of $25 (platform requirement)
    initial_deposit = np.where(
        opened_savings_account == 1,
        np.maximum(
            25,  # minimum deposit
            np.random.lognormal(
                mean=np.log(deposit_mean),
                sigma=np.log(deposit_std / deposit_mean + 1),
                size=n_users
            )
        ),
        np.nan  # No deposit if didn't open account
    )
    
    # GUARDRAIL METRICS
    
    # 1. App engagement (7-day return rate) - slightly lower for treatment due to feature complexity
    engagement_prob = BASELINE_APP_ENGAGEMENT if group_name == 'control' else BASELINE_APP_ENGAGEMENT - 0.02
    returned_to_app_7d = np.random.binomial(1, engagement_prob, n_users)
    
    # 2. Account abandonment (opened but never deposited again) - slightly higher for treatment (bonus chasers)
    abandonment_prob = BASELINE_ABANDONMENT_RATE if group_name == 'control' else BASELINE_ABANDONMENT_RATE + 0.03
    # Only applies to users who opened accounts
    account_abandoned = np.where(
        opened_savings_account == 1,
        np.random.binomial(1, abandonment_prob, n_users),
        np.nan
    )
    
    # 3. Support tickets (treatment may cause confusion, increasing tickets)
    support_tickets_prob = BASELINE_SUPPORT_TICKETS_PER_1000 / 1000
    if group_name == 'treatment':
        support_tickets_prob *= 1.25  # 25% increase in support volume
    filed_support_ticket = np.random.binomial(1, support_tickets_prob, n_users)
    
    # 4. Transaction account activity (ensure savings doesn't cannibalize checking)
    transaction_activity_prob = BASELINE_TRANSACTION_ACTIVITY
    used_transaction_account = np.random.binomial(1, transaction_activity_prob, n_users)
    
    # Create DataFrame
    df = pd.DataFrame({
        'user_id': user_ids,
        'experiment_group': experiment_group,
        'exposure_date': exposure_dates,
        'opened_savings_account': opened_savings_account,
        'initial_deposit_amount': initial_deposit,
        'returned_to_app_7d': returned_to_app_7d,
        'account_abandoned': account_abandoned,
        'filed_support_ticket': filed_support_ticket,
        'used_transaction_account': used_transaction_account
    })
    
    return df

# ============================================================================
# GENERATE FULL DATASET
# ============================================================================

print("Generating synthetic A/B test data for Nclusion savings incentive experiment...")
print(f"Control group size: {N_CONTROL:,}")
print(f"Treatment group size: {N_TREATMENT:,}")
print(f"Total sample size: {N_TOTAL:,}\n")

# Generate control group
df_control = generate_user_cohort(
    n_users=N_CONTROL,
    group_name='control',
    conversion_rate=BASELINE_CONVERSION_RATE,
    deposit_mean=CONTROL_DEPOSIT_MEAN,
    deposit_std=CONTROL_DEPOSIT_STD
)

# Generate treatment group
df_treatment = generate_user_cohort(
    n_users=N_TREATMENT,
    group_name='treatment',
    conversion_rate=TREATMENT_CONVERSION_RATE,
    deposit_mean=TREATMENT_DEPOSIT_MEAN,
    deposit_std=TREATMENT_DEPOSIT_STD
)

# Combine into single dataset
df_experiment = pd.concat([df_control, df_treatment], ignore_index=True)

# Shuffle rows (simulate random assignment order)
df_experiment = df_experiment.sample(frac=1, random_state=42).reset_index(drop=True)

# ============================================================================
# SAVE DATA
# ============================================================================

output_file = '/Users/maxvargas/product_analytics_lab/data/nclusion_ab_test_data.csv'
df_experiment.to_csv(output_file, index=False)

print(f"✅ Dataset generated and saved to '{output_file}'")
print(f"\nDataset shape: {df_experiment.shape}")
print(f"\nFirst few rows:\n")
print(df_experiment.head(10))

# ============================================================================
# DATA SUMMARY
# ============================================================================

print("\n" + "="*80)
print("EXPERIMENT SUMMARY")
print("="*80)

# Group sizes
print("\n📊 Group Distribution:")
print(df_experiment['experiment_group'].value_counts())

# Primary metric (conversion rate) by group
print("\n🎯 PRIMARY METRIC: Savings Account Conversion Rate")
conversion_by_group = df_experiment.groupby('experiment_group')['opened_savings_account'].agg(['sum', 'mean', 'count'])
conversion_by_group.columns = ['total_conversions', 'conversion_rate', 'total_users']
print(conversion_by_group)

# Secondary metric (initial deposit) by group (only for converters)
print("\n💰 SECONDARY METRIC: Initial Deposit Amount (Converters Only)")
deposit_by_group = df_experiment[df_experiment['opened_savings_account'] == 1].groupby('experiment_group')['initial_deposit_amount'].agg(['mean', 'median', 'std', 'count'])
deposit_by_group.columns = ['mean_deposit', 'median_deposit', 'std_deposit', 'n_converters']
print(deposit_by_group)

# Guardrail metrics
print("\n🛡️  GUARDRAIL METRICS:")
guardrails = df_experiment.groupby('experiment_group').agg({
    'returned_to_app_7d': 'mean',
    'filed_support_ticket': 'mean',
    'used_transaction_account': 'mean'
}).round(4)
guardrails.columns = ['app_engagement_rate', 'support_ticket_rate', 'transaction_activity_rate']
print(guardrails)

# Account abandonment (only for converters)
print("\n🚨 Account Abandonment Rate (Converters Only):")
abandonment_by_group = df_experiment[df_experiment['opened_savings_account'] == 1].groupby('experiment_group')['account_abandoned'].mean()
print(abandonment_by_group.round(4))

print("\n" + "="*80)
print("Data generation complete! Ready for analysis.")
print("="*80)