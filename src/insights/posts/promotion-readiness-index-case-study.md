---
layout: post.njk
title: Tenure and Performance Ratings Are Weak Predictors of Who Gets Promoted
date: 2026-06-18
excerpt: A synthetic-data walkthrough of building a promotion-readiness model — and why the two inputs most companies rely on turned out to be the weakest ones in it.
tags:
  - posts
  - prediction
---

Most organizations decide who's ready for promotion using two inputs: how long someone has been in their role, and their last performance rating. Both are convenient. Neither was designed to predict readiness — and in a demo analysis I ran using synthetic data, a model built on just those two barely beat a coin flip at identifying who actually got promoted within 18 months (AUC 0.57 on held-out data).

Replacing them with a structured behavioral observation rubric and a validated assessment — while still controlling for tenure and performance rating — lifted discrimination to 0.67, with the assessment data emerging as the strongest, most statistically reliable predictor in the model. Tenure, once the better data was in the model, added almost nothing.

The full write-up walks through the measurement design, the model, an honest look at where its calibration wobbles (small holdout cells produce noisy estimates, and the write-up says so rather than smoothing over it), and what a promotion-readiness score actually looks like translated into decision-usable tiers.

[Read the full case study &rarr;](/case-studies/promotion-readiness-index/)
