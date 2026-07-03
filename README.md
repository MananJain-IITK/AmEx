# American Express Campus Challenge 2026 — Round 1
## Profitability Framework (V16)

## Project Description
Built a deterministic profitability framework to identify the Top 20% most profitable cardmembers out of a 500,000-customer Premier Card portfolio. The task provided 23 anonymized features (`f1`–`f23`) spanning revenue, cost, engagement, and risk signals, with the objective of designing an interpretable, business-grounded equation — not a black-box model — to rank-order cardmembers by estimated annual net profitability.

## Approach
Rather than training a predictive ML model directly on the labels, we took a business-logic-first approach:

1. **De-anonymization** — mapped each masked feature to a real-world driver (revolving balance, spend categories, risk score, benefit usage, etc.) using the Premier Card product brief and cross-referencing against American Express's 2023 10-K disclosures for realistic coefficients.
2. **Additive P&L construction** — built the profitability score as a transparent sum of revenue and cost line items, mirroring how an issuer would actually attribute per-customer P&L.
3. **Observed vs. estimated rewards liability (V16 change)** — the initial version estimated points cost from spend categories × reward multipliers × redemption rate. V16 replaces this with `f21` (actual points redeemed), pricing the rewards liability off *observed* redemption behavior rather than a population-average assumption. This is a more direct measure of realized cost to the issuer.
4. **ML benchmarking** — validated the linear formula against a self-supervised Logistic Regression and Gradient Boosted Tree trained on the framework's own output, confirming the linear structure captures the large majority of the achievable signal (>98% overlap).
5. **Percentile-rank output scaling (V16 change)** — final predictions are converted from raw dollar scores to a `[0, 1]` percentile rank, which preserves rank order exactly (no impact on Top-20% accuracy) while being robust to outlier customers with unusually large raw scores.

## Results
- **Leaderboard accuracy: 0.866**
- Percentile-rank scaling ensures the exported prediction column is outlier-robust and directly interpretable as "more profitable than X% of the portfolio."

## Key Insights
1. **Travel spend margin paradox** — pure transactors maximizing 5x travel-category spend (airline/lodging) are net-negative for the issuer once points liability is netted against interchange revenue (≈ -0.36¢ per dollar spent).
2. **"Safe revolver" profile** — the highest-profitability customers are not the highest spenders, but those carrying a large revolving balance with a low probability of default; net interest income dominates interchange for this segment.
3. **Downside risk dominates upside engagement** — collection and retention costs penalize lifetime value far more heavily than any engagement signal (logins, email opens) helps it.
4. **Linear structure ceiling** — benchmarking against GBT suggests the additive formula is near the practical ceiling for a linear structure; remaining variance likely reflects non-linear caps (e.g. points cost naturally bounded by interchange revenue) that a purely additive model can't express.

## The Framework

### Variables Used
- **Revenue drivers:** `f1` (Revolving Balance), `f6` (Airline Spend), `f7` (Other Spend), `f8` (Entertainment Spend), `f9` (Lodging Spend), `f10` (Dining Spend), `f17` (Total Lend Line), `f19` (Supplementary Accounts), `f20` (Active Charge Cards)
- **Cost drivers:** `f2` (Cancellation Calls), `f3` (Cancellation Calls due to Collection), `f11` (Risk Score), `f13` (Lounge Access Count), `f14` (Airline Credits Used), `f15` (Cab Benefits Usage), `f16` (Entertainment Credit Used), `f21` (Rewards Points Redeemed)
- **Excluded after testing:** `f4` (unredeemed points balance), `f5` (total spend — redundant with subcategories), `f12` (logins), `f18` (consumer lend line), `f22`/`f23` (email engagement)

### Profitability Equation
```
Profitability = R_Interchange + R_Interest + R_Supp + R_CreditLine
                 - C_PointsRedeemed - C_Lounge - C_Airline - C_Cab
                 - C_Entertainment - C_ECL - C_Retention
```

| Term | Formula |
|---|---|
| `R_Interchange` | `(f6+f9)*0.030 + (f7+f8+f10)*0.020` |
| `R_Interest` | `f1 * 0.24` |
| `R_Supp` | `f19*100 + f20*100` |
| `R_CreditLine` | `f17 * 0.001` |
| `C_PointsRedeemed` | `f21 * 0.007` |
| `C_Lounge` | `f13 * 40` |
| `C_Airline` | `f14` |
| `C_Cab` | `f15 * 15` |
| `C_Entertainment` | `f16` |
| `C_ECL` | `f1*f11 + f3*1000 + f3*f1` |
| `C_Retention` | `f2 * 300` |

### Prediction Logic
Score every cardmember using the equation above, then convert to a percentile rank via `score.rank(pct=True)`. Customers above the 0.80 percentile are classified as Top 20% most profitable.

### Coefficient / Weight Derivation
| Parameter | Value | Source |
|---|---|---|
| Interchange (5x categories) | 3.0% | Amex disclosed merchant discount rate range |
| Interchange (1x categories) | 2.0% | Amex disclosed merchant discount rate range |
| Revolving APR | 24% | Standard Premier product APR |
| Points WAC | 0.7¢/point | Amex 2023 10-K |
| Lounge cost | $40/visit | Priority Pass / Centurion industry benchmark |
| Collection penalty | $1,000 + full balance | Standard LGD assumption for charge-offs |
| Retention cost | $300/call | Proxy for average retention offer value |
| Supplementary account value | $100/account | Proxy for annual fee contribution |

### Assumptions
1. Points WAC of 0.7¢/point, applied to *actual redeemed* points rather than earned points.
2. Interchange: 3.0% on 5x categories, 2.0% on 1x categories.
3. Revolving APR: 24% gross.
4. LGD: 100% of revolving balance lost on charge-off.
5. Missing `f11` imputed at population median; spend/benefit fields imputed as 0.
6. Annual fee treated as a constant across all cardmembers and excluded from rank ordering.
7. Percentile-rank scaling is monotonic and does not alter rank order or Top-20% classification.

### Validation Approach
1. Business-logic cross-check against Amex's 2023 10-K and the Premier Card product brief.
2. Correlation diagnostics confirming `f21` (actual redemption) is not redundant with the spend-derived earned-points estimate, supporting its use as independent cost signal.
3. Top-20% overlap analysis between formula variants to isolate which changes meaningfully re-rank customers near the decision boundary.
4. ML benchmarking (Logistic Regression, GBT) against the framework's own output to confirm the linear structure is near its achievable ceiling.
5. Iterative leaderboard submission testing.

## Repository / Files
- `v16_experiments.py` — script that builds the baseline and candidate score variants, runs correlation and Top-20% overlap diagnostics, and exports submission files with percentile-rank-scaled predictions.
- `submission_v16a_redeemed_pts.xlsx` — primary V16 submission (redemption-based points cost).

## How to Run
1. Update `DATA_PATH`, `OUT_DIR`, and `TEMPLATE_PATH` in `v16_experiments.py` to point to your local files.
2. Run:
   ```bash
   python v16_experiments.py
   ```
3. Submission files are written to `OUT_DIR`, ready for upload to the Unstop submission portal.

## Next Steps (Not Yet Applied)
- Cap points cost per spend category at that category's interchange revenue (structural cap rather than unbounded linear penalty).
- Winsorize `f1` and spend fields at the 1st/99th percentile to reduce outlier influence on rank order.
- Cap total ECL at the outstanding balance (`f1`) to avoid double-penalizing high-`f3` customers.
- Net entertainment spend against entertainment credit used (`f8` vs. `f16`) to avoid crediting and penalizing the same dollars.