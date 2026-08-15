import pandas as pd

df1 = pd.read_csv('outputs/reproducibility/balanced_run1.csv')
df3 = pd.read_csv('outputs/reproducibility/balanced_run3.csv')

merged = df1.merge(df3, on='clip_name', suffixes=('_run1', '_run3'))
merged['same'] = merged['prediction_run1'] == merged['prediction_run3']
merged['same_reason'] = merged['reason_run1'] == merged['reason_run3']

print('| Clip | Run 1 | Run 3 | Same? |')
print('|:---|:---:|:---:|:---:|')
for _, r in merged.iterrows():
    c = r['clip_name']
    p1 = r['prediction_run1']
    p3 = r['prediction_run3']
    s = 'YES' if r['same'] else '**NO**'
    print(f'| `{c}` | `{p1}` | `{p3}` | {s} |')

print('\n=== SUMMARY ===')
n_diff_pred = sum(~merged['same'])
n_diff_votes = sum(~merged['same_reason'])
print(f'Total clips: {len(merged)}')
print(f'Predictions identical: {len(merged) - n_diff_pred} / {len(merged)}')
print(f'Changed predictions: {n_diff_pred}')
print(f'Changed vote patterns: {n_diff_votes}')
