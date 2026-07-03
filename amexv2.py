import pandas as pd
import os

BASE = '/home/manan/AmEx'
DATA_PATH = f'{BASE}/r1.csv'
OUT_DIR = f'{BASE}/output'
TEMPLATE_PATH = f'{BASE}/6a3cb64c7cae4_campus_challenge_r1_submission_template.xlsx'

df = pd.read_csv(DATA_PATH)
df_framework = pd.read_excel(TEMPLATE_PATH, sheet_name='Profitability Framework')

# Imputation
risk_median = df['f11'].median()
df['f11'] = df['f11'].fillna(risk_median)
for f in ['f6', 'f7', 'f8', 'f9', 'f10']:
    df[f] = df[f].fillna(0)
for f in ['f1', 'f2', 'f3', 'f4', 'f5', 'f12', 'f13', 'f14', 'f15',
          'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23']:
    df[f] = df[f].fillna(0)

# === v17: V14 + Net Interest Margin refinement ===
# Insight: Amex earns 24% gross APR but has a "Cost of Funds" (cost to borrow the money)
# For premium cardmembers, Amex's cost of funds in 2023 was ~4-5% (Fed Funds Rate era)
# So Net Interest Margin = 24% APR - 5% Cost of Funds = ~19% NIM
# BUT: the ECL (expected credit loss) also eats into this
# Standard banking formula: NIM = Gross APR - Cost of Funds - Provision for Credit Losses
# Our current formula handles ECL separately so we should use GROSS APR (24%) not NIM
# 
# THE REAL OPPORTUNITY: f17 (Lending Credit Line) vs f18 (Charge Card Line)
# f17 = lending credit line — amount Amex is exposed to on revolving products
# f18 = charge card line — effectively unlimited but proxy for creditworthiness
# Currently: R_credit_line = f17 * 0.0001 (tiny contribution)
# 
# ACTUAL INSIGHT from marginal analysis:
# Marginal IN customers: f17=10,899 (higher credit line)
# Marginal OUT customers: f17=10,483 (lower credit line)
# f17 difference = 415 — meaningful at the margin!
#
# What if f17 signals creditworthiness and we should weight it more?
# A higher credit line = Amex is comfortable extending more credit = less risky = more profitable

R_interchange = (df['f6'] + df['f9']) * 0.030 + (df['f7'] + df['f8'] + df['f10']) * 0.020
R_interest = df['f1'] * 0.24
R_supp = df['f19'] * 100.0 + df['f20'] * 100.0

# CHANGE from V14: Increase f17 credit line weight from 0.0001 to 0.001
# (Higher credit line customers are more creditworthy, commanding a 10x stronger signal)
R_credit_line = df['f17'] * 0.001

pts_earned = df['f6'] * 5 + df['f9'] * 5 + df['f7'] * 1 + df['f8'] * 1 + df['f10'] * 1
C_pts_accrual = pts_earned * 0.007 * 0.96   # Amex 10-K: WAC=0.7¢, URR=96%

C_lounge = df['f13'] * 40.0
C_airline = df['f14']
C_cab = df['f15'] * 15.0
C_ent = df['f16']
C_ecl = df['f1'] * df['f11'] * 1.0 + (df['f3'] * 1000.0) + (df['f3'] * df['f1'] * 1.0)
C_retention = df['f2'] * 300.0

df['Profitability_Score'] = (
    R_interchange + R_interest + R_supp + R_credit_line
    - C_pts_accrual - C_lounge - C_airline - C_cab - C_ent - C_ecl - C_retention
)

sub = df[['id', 'Profitability_Score']].copy()
sub = sub.rename(columns={'id': 'ID', 'Profitability_Score': 'Prediction'})
sub = sub.sort_values('ID').reset_index(drop=True)

xlsx_path = f'{OUT_DIR}/submission_v17_credit_line_signal.xlsx'
with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
    sub.to_excel(writer, sheet_name='Predictions', index=False)
    df_framework.to_excel(writer, sheet_name='Profitability Framework', index=False)

# Verify
xl_out = pd.ExcelFile(xlsx_path)
print("=== VERIFICATION ===")
print(f"Sheets: {xl_out.sheet_names}")
for sheet in xl_out.sheet_names:
    df_chk = pd.read_excel(xlsx_path, sheet_name=sheet)
    print(f"  '{sheet}' -> cols={list(df_chk.columns)}, rows={len(df_chk)}")
print(f"\nGenerated: {xlsx_path}")