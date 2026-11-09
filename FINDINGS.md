# Findings - BizForge (Telco churn)

## Question
Which customers are likely to leave, how well can we rank them, and does targeting them pay off?

## Data
IBM Telco Customer Churn sample: 7,043 customers, 26.5% churned. Not a random sample of a real operator, so absolute rates should not be generalised.

## What the data says
- Contract type dominates: month-to-month churn is 42.7%, one-year 11.3%, two-year 2.8%.
- Churn falls steeply with tenure: 54% in the first 6 months, 9.6% beyond 48 months.
- Permutation importance (held-out AUC drop) agrees: month-to-month contract (0.056) and tenure (0.046) matter far more than everything else; the next feature (fiber internet) is ~0.01.

## Model results (test set, n=1,409)
- XGBoost, logistic regression and random forest are statistically indistinguishable in CV (AUC 0.844-0.848, fold std ~0.011-0.013). XGBoost was selected by a margin smaller than the noise. A regularised logistic regression is almost as good and far easier to explain; for deployment I would choose it.
- Test AUC 0.847 (95% CI 0.825-0.868). Calibrated probabilities track actual churn across deciles (see `reports/calibration.csv`), so they can be used for expected-value calculations, not just ranking.
- Top 10% riskiest customers churn at 2.9x the base rate and contain 29% of all churners.

## Campaign analysis (scenario, not measurement)
With an assumed 30% save rate, $25 per contact and 12 months of revenue credited per save, contacting the riskiest ~64% maximises expected net value (~$368k vs ~$324k for contacting everyone). When contact cost is high or the save rate low, targeting matters much more: at 10% save rate and $25/contact, contacting everyone loses money while targeting the top 36% earns ~$66k (`reports/campaign_sensitivity.csv`). The real uncertainty is the save rate, which this data cannot estimate.

## Limitations
- Associational, not causal: contract type predicts churn but changing a customer's contract is not equivalent to the observed pattern.
- No time dimension: one snapshot, so no out-of-time validation or drift analysis.
- Dataset is a public sample; results show method, not a specific business outcome.

## What I would do next
Collect a randomized retention-offer test to estimate uplift per segment (uplift modelling), add a time-based split with a dated dataset, and compare against a simple rules baseline (month-to-month and tenure < 12 months).
