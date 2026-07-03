---
layout: post.njk
title: Your Engagement Survey Is a Decent Turnover Predictor. It's Not Enough.
date: 2026-07-02
excerpt: A synthetic-data walkthrough of building a turnover risk score — and why the metric most companies already have needs company, not replacement.
tags:
  - posts
  - prediction
---

Most turnover prediction stops at the engagement survey: low scorers get flagged, everyone else gets ignored. In a demo analysis using synthetic data, engagement score alone actually held up reasonably well — a holdout AUC of 0.66, real signal, not noise.

Adding three inputs the engagement survey can't see — pay competitiveness, time since last promotion, and the share of an employee's immediate team that has recently left ("turnover contagion," a documented effect in the turnover literature) — lifted holdout AUC to 0.77. The clearest finding: every 10-point rise in a team's trailing attrition rate raised an individual's own odds of leaving by roughly 49%. Turnover spreads through a team, not just through individual dissatisfaction.

The full write-up covers the model, an honest look at where the risk-decile ranking wobbles in a small holdout sample, and the practical payoff: the top 20% of the workforce by predicted risk captures half of all actual departures.

[Read the full case study &rarr;](/case-studies/turnover-risk-score/)
