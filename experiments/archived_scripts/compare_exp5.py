import pandas as pd
import numpy as np

df_gt = pd.read_csv('data/ground_truth.csv')
df_gt = df_gt[df_gt['is_evaluated'] == True].copy()

df_v1 = pd.read_csv('outputs/predictions/predictions_20260814_091949.csv')
df_temp = pd.read_csv('outputs/predictions/predictions_20260814_103316.csv')

merged = df_gt[['clip_name', 'ground_truth']].merge(
    df_v1[['clip_name', 'prediction', 'reason']].rename(columns={'prediction': 'pred_v1'}),
    on='clip_name'
).merge(
    df_temp[['clip_name', 'prediction']].rename(columns={'prediction': 'pred_temp'}),
    on='clip_name'
)

merged['pred_policy_a'] = np.where((merged['pred_v1'] == 'jaywalking') & (merged['pred_temp'] == 'jaywalking'), 'jaywalking', 'compliant')
merged['pred_policy_b'] = np.where(merged['pred_v1'] == 'compliant', 'compliant', merged['pred_temp'])
merged['pred_policy_c'] = np.where((merged['pred_temp'] == 'jaywalking') & (merged['pred_v1'] == 'jaywalking'), 'jaywalking', 'compliant')

print('| Clip | Ground Truth | V1 (Baseline) | Temporal (Exp 3) | Policy A (Consensus) | Policy B (V1 Priority) | Policy C (Confirmation) |')
print('|:---|:---:|:---:|:---:|:---:|:---:|:---:|')
for idx, r in merged.iterrows():
    c = r['clip_name']
    gt = r['ground_truth']
    v1 = r['pred_v1']
    temp = r['pred_temp']
    pa = r['pred_policy_a']
    pb = r['pred_policy_b']
    pc = r['pred_policy_c']
    print(f'| `{c}` | `{gt}` | `{v1}` | `{temp}` | `{pa}` | `{pb}` | `{pc}` |')
