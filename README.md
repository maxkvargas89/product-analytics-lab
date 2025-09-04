#  Product Analytics Lab

This repository is my sandbox for practicing product analysis, experimentation, and statistical evaluation. Using synthetic or real product usage data, I explore A/B testing, retention modeling, feature impact analysis, and other techniques that help drive product decisions.

---

##  Objectives

- Execute rigorous **A/B testing** to estimate treatment effects.
- Analyze **user retention patterns** and factors influencing churn.
- Model **conversion funnels**, **time-to-event**, and **growth metrics**.
- Practice **causal inference**: difference-in-differences, regression discontinuity designs.
- Build dashboards and visualizations supporting hypothesis-driven product decisions.

---

##  What’s Inside

| Folder / File        | Description |
|----------------------|-------------|
| `data/`              | Raw or simulated datasets (events, sessions, conversions, cohorts) |
| `scripts/`           | Reusable modules for modeling, simulation, and metrics pipelines |
| `README.md`          | Overview and documentation for this repository |

---

##  Analytics & Experimentation Techniques Practiced

- **A/B Testing & Significance Analysis**  
  - Compute test statistics, p-values, and confidence intervals.
  - Adjust for multiple hypothesis testing or sequential tests.

- **Retention & Churn Modeling**  
  - Kaplan-Meier curves, survival analysis, cohort-based retention visuals.
  - Identify behavioral predictors of churn via logistic regression.

- **Conversion Funnel & Time-To-Event Modeling**  
  - Visualize funnel drop-offs and time lag between stages.
  - Use hazard models for time-to-conversion dynamics.

- **Causal Analysis Approaches**  
  - Define treatment/control groups; simulate assignment.
  - Apply **difference-in-differences** and **regression discontinuity** where appropriate.

- **Feature Impact Modeling**  
  - Use regression or uplift models to estimate lift from new features or campaigns.
