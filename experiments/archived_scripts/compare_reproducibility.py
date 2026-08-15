import pandas as pd

df1 = pd.read_csv('outputs/reproducibility/balanced_run1.csv')
df2 = pd.read_csv('outputs/reproducibility/balanced_run2.csv')

merged = df1.merge(df2, on='clip_name', suffixes=('_run1', '_run2'))
merged['same'] = merged['prediction_run1'] == merged['prediction_run2']
merged['same_reason'] = merged['reason_run1'] == merged['reason_run2']

print('| Clip | Run 1 | Run 2 | Same? |')
print('|:---|:---:|:---:|:---:|')
for _, r in merged.iterrows():
    c = r['clip_name']
    p1 = r['prediction_run1']
    p2 = r['prediction_run2']
    s = 'YES' if r['same'] else '**NO**'
    print(f'| `{c}` | `{p1}` | `{p2}` | {s} |')

print('\n=== SUMMARY ===')
n_diff_pred = sum(~merged['same'])
n_diff_votes = sum(~merged['same_reason'])
print(f'Total clips: {len(merged)}')
print(f'Predictions identical: {len(merged) - n_diff_pred} / {len(merged)}')
print(f'Changed predictions: {n_diff_pred}')
print(f'Changed vote patterns: {n_diff_votes}')

for _, r in merged[~merged['same_reason']].iterrows():
    print(f"- `{r['clip_name']}` (GT: {r['ground_truth_run1']}): Run 1 = {r['prediction_run1']} ({r['reason_run1']}) vs Run 2 = {r['prediction_run2']} ({r['reason_run2']})")
