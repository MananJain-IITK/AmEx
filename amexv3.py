"""
Amex Campus Challenge - V19 Experiments
=========================================
Goal: test whether f21 (actual redeemed points), f18 (consumer lend line),
and f4 (unredeemed points balance) improve on the V15 baseline formula.

Update the paths below to match your local setup, then run:
    python v19_experiments.py
"""

import pandas as pd
import numpy as np
import os

BASE = '/home/manan/AmEx'
DATA_PATH = f'{BASE}/r1.csv'
OUT_DIR = f'{BASE}/output'
TEMPLATE_PATH = f'{BASE}/6a3cb64c7cae4_campus_challenge_r1_submission_template.xlsx'

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df_framework = pd.read_excel(TEMPLATE_PATH, sheet_name='Profitability Framework')

# ----------------------------------------------------------------------
# 1. Imputation (same as V15)
# ----------------------------------------------------------------------
risk_median = df['f11'].median()
df['f11'] = df['f11'].fillna(risk_median)
for f in ['f6', 'f7', 'f8', 'f9', 'f10']:
    df[f] = df[f].fillna(0)
for f in ['f1', 'f2', 'f3', 'f4', 'f5', 'f12', 'f13', 'f14', 'f15',
          'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23']:
    df[f] = df[f].fillna(0)

# ----------------------------------------------------------------------
# 2. Baseline V15 score (for comparison)
# ----------------------------------------------------------------------
R_interchange = (df['f6'] + df['f9']) * 0.030 + (df['f7'] + df['f8'] + df['f10']) * 0.020
R_interest = df['f1'] * 0.24
R_supp = df['f19'] * 100.0 + df['f20'] * 100.0
R_credit_line = df['f17'] * 0.001

pts_earned = df['f6'] * 5 + df['f9'] * 5 + df['f7'] * 1 + df['f8'] * 1 + df['f10'] * 1
C_pts_earned_based = pts_earned * 0.007 * 0.96   # V15 method: estimated from spend

C_lounge = df['f13'] * 40.0
C_airline = df['f14']
C_cab = df['f15'] * 15.0
C_ent = df['f16']
C_ecl = df['f1'] * df['f11'] * 1.0 + (df['f3'] * 1000.0) + (df['f3'] * df['f1'] * 1.0)
C_retention = df['f2'] * 300.0

FIXED_TERMS = (
    R_interchange + R_interest + R_supp + R_credit_line
    - C_lounge - C_airline - C_cab - C_ent - C_ecl - C_retention
)

df['score_v15'] = FIXED_TERMS - C_pts_earned_based
df['score_v15_scaled_0_1'] = df['score_v15'].rank(pct=True)

# ----------------------------------------------------------------------
# 3. Diagnostic: correlations of the unused features
# ----------------------------------------------------------------------
print("=== Correlation checks for unused/underused features ===\n")

print("f21 (redeemed pts) vs pts_earned (spend-derived estimate):",
      round(df['f21'].corr(pts_earned), 3))
print("f21 (redeemed pts) vs V15 score:",
      round(df['f21'].corr(df['score_v15']), 3))
print("f4  (unredeemed pts balance) vs V15 score:",
      round(df['f4'].corr(df['score_v15']), 3))
print("f18 (consumer lend line) vs f17 (total lend line):",
      round(df['f18'].corr(df['f17']), 3))
print("f18 (consumer lend line) vs V15 score:",
      round(df['f18'].corr(df['score_v15']), 3))
print()

# ----------------------------------------------------------------------
# 4. Candidate V19 variants
# ----------------------------------------------------------------------
variants = {}

# V19a: swap earned-points cost for ACTUAL redeemed-points cost
C_pts_redeemed_based = df['f21'] * 0.007
variants['v19a_redeemed_pts'] = FIXED_TERMS - C_pts_redeemed_based

# V19b: blend earned + redeemed (average of the two liability views)
C_pts_blended = 0.5 * C_pts_earned_based + 0.5 * C_pts_redeemed_based
variants['v19b_blended_pts'] = FIXED_TERMS - C_pts_blended

# V19c: v19a + forward liability discount from unredeemed balance (f4)
C_pts_overhang = df['f4'] * 0.007 * 0.10   # small forward-looking discount, tune this
variants['v19c_redeemed_plus_overhang'] = (FIXED_TERMS - C_pts_redeemed_based
                                            - C_pts_overhang)

# V19d: v19a + f18 as additional creditworthiness revenue term
R_consumer_credit = df['f18'] * 0.0005     # tune this coefficient
variants['v19d_redeemed_plus_f18'] = (FIXED_TERMS - C_pts_redeemed_based
                                       + R_consumer_credit)

# V19e: dining (f10) given its own multiplier bucket instead of lumped 1x
#       re-derive R_interchange with f10 broken out at an assumed elevated rate
R_interchange_dining = (
    (df['f6'] + df['f9']) * 0.030          # 5x categories: airline, lodging
    + df['f10'] * 0.025                     # dining treated separately, slightly elevated
    + (df['f7'] + df['f8']) * 0.020         # remaining 1x categories
)
pts_earned_dining = df['f6']*5 + df['f9']*5 + df['f10']*3 + df['f7']*1 + df['f8']*1
C_pts_dining_based = pts_earned_dining * 0.007 * 0.96
fixed_terms_dining = (
    R_interchange_dining + R_interest + R_supp + R_credit_line
    - C_lounge - C_airline - C_cab - C_ent - C_ecl - C_retention
)
variants['v19e_dining_multiplier'] = fixed_terms_dining - C_pts_dining_based

# ----------------------------------------------------------------------
# 5. Compare each variant's Top-20% overlap with the V15 baseline
#    (a large overlap = little change; a smaller overlap = variant is
#     meaningfully re-ranking people, which is where accuracy gains
#     or losses will come from)
# ----------------------------------------------------------------------
n = len(df)
top20_cutoff = int(n * 0.20)

baseline_top20 = set(df.sort_values('score_v15', ascending=False).head(top20_cutoff).index)

print("=== Top-20% overlap of each variant vs V15 baseline ===\n")
for name, score in variants.items():
    df[name] = score
    variant_top20 = set(df.sort_values(name, ascending=False).head(top20_cutoff).index)
    overlap = len(baseline_top20 & variant_top20) / top20_cutoff
    print(f"{name:35s} overlap={overlap:.4f}  "
          f"(spearman corr={df['score_v15'].corr(df[name], method='spearman'):.4f})")

print("""
Next step:
Submit v19a (and whichever of v19c/v19d/v19e shows the lowest overlap
with the baseline, since that's the one most likely to change your
leaderboard accuracy) and compare actual scores. Since you only have
10 submissions, prioritize v19a first — it's the highest-confidence
change since f21 is *observed* redemption, not an estimate.
""")

# ----------------------------------------------------------------------
# 6. Write out submission files for the top candidate(s)
#    Predictions are rescaled to [0, 1] via PERCENTILE RANK rather than
#    min-max. This is still a monotonic transform (rank order, and
#    therefore Top-20% accuracy, is IDENTICAL to the raw dollar score),
#    but it's far more robust to outliers than min-max: a single whale
#    customer with an extreme raw score won't compress everyone else's
#    scaled values into a tiny sliver near 0. Each value becomes
#    "this customer is more profitable than X% of the population."
# ----------------------------------------------------------------------
def percentile_scale(series):
    return series.rank(pct=True)

def write_submission(score_col, out_name):
    sub = df[['id', score_col]].copy()
    sub[score_col] = percentile_scale(sub[score_col])
    sub = sub.rename(columns={'id': 'ID', score_col: 'Prediction'})
    sub = sub.sort_values('ID').reset_index(drop=True)
    path = f'{OUT_DIR}/{out_name}.xlsx'
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        sub.to_excel(writer, sheet_name='Predictions', index=False)
        df_framework.to_excel(writer, sheet_name='Profitability Framework', index=False)
    print(f"Wrote {path}")
    return path

write_submission('v19a_redeemed_pts', 'submission_v19a_redeemed_pts')
write_submission('v19c_redeemed_plus_overhang', 'submission_v19c_overhang')
write_submission('v19d_redeemed_plus_f18', 'submission_v19d_f18')
write_submission('v19e_dining_multiplier', 'submission_v19e_dining')