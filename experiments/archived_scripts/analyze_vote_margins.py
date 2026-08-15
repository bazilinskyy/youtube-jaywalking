import json
import pandas as pd
import numpy as np

# Load ground truth and V1 predictions
df_gt = pd.read_csv('data/ground_truth.csv')
df_gt = df_gt[df_gt['is_evaluated'] == True].copy()

df_v1 = pd.read_csv('outputs/predictions/predictions_20260814_091949.csv')

def extract_jw_votes(reason):
    if '3 jaywalking' in reason:
        return 3
    elif '2 jaywalking' in reason:
        return 2
    elif '1 jaywalking' in reason:
        return 1
    elif '0 jaywalking' in reason:
        return 0
    return 0

df_v1['jw_votes'] = df_v1['reason'].apply(extract_jw_votes)

merged = df_gt[['clip_name', 'ground_truth']].merge(
    df_v1[['clip_name', 'prediction', 'confidence', 'reason', 'jw_votes']],
    on='clip_name'
)

print('=== 1. VOTE PATTERN BREAKDOWN TABLE ===')
patterns = [3, 2, 1, 0]
pattern_names = {
    3: '3/3 JAYWALKING (Unanimous Violation)',
    2: '2/3 JAYWALKING (Split Violation)',
    1: '1/3 JAYWALKING (Split Compliant)',
    0: '0/3 JAYWALKING (Unanimous Compliant)'
}

pattern_stats = []
for p in patterns:
    sub = merged[merged['jw_votes'] == p]
    n_clips = len(sub)
    n_jw = sum(sub['ground_truth'] == 'jaywalking')
    n_comp = sum(sub['ground_truth'] == 'compliant')
    
    # If this bucket is classified as JAYWALKING (for 3 and 2) vs COMPLIANT (for 1 and 0):
    default_pred = 'jaywalking' if p >= 2 else 'compliant'
    correct = sum(sub['ground_truth'] == default_pred)
    incorrect = n_clips - correct
    
    # Empirical probability / Precision of Jaywalking in this bucket:
    jw_prob = (n_jw / n_clips * 100) if n_clips > 0 else 0
    comp_prob = (n_comp / n_clips * 100) if n_clips > 0 else 0
    
    pattern_stats.append({
        'Vote Pattern': pattern_names[p],
        'Clips (N)': n_clips,
        'GT Jaywalking': n_jw,
        'GT Compliant': n_comp,
        'V1 Pred': default_pred.upper(),
        'Correct': correct,
        'Incorrect': incorrect,
        'P(Jaywalking)': f'{jw_prob:.1f}%',
        'P(Compliant)': f'{comp_prob:.1f}%'
    })

print(pd.DataFrame(pattern_stats).to_string(index=False))

print('\n=== 2. THRESHOLD POLICIES EVALUATION ===')
def calc_metrics(y_true, y_pred):
    tp = sum((y_true == 'jaywalking') & (y_pred == 'jaywalking'))
    tn = sum((y_true == 'compliant') & (y_pred == 'compliant'))
    fp = sum((y_true == 'compliant') & (y_pred == 'jaywalking'))
    fn = sum((y_true == 'jaywalking') & (y_pred == 'compliant'))
    total = len(y_true)
    acc = (tp + tn) / total * 100
    prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return {
        'Accuracy': f'{acc:.2f}%',
        'Precision': f'{prec:.2f}%',
        'Recall': f'{rec:.2f}%',
        'Specificity': f'{spec:.2f}%',
        'F1 Score': f'{f1:.2f}%',
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn
    }

thresholds = [
    ('Threshold >= 1/3 (Sensitive)', merged['jw_votes'] >= 1),
    ('Threshold >= 2/3 (Baseline V1 / Majority)', merged['jw_votes'] >= 2),
    ('Threshold = 3/3 (Unanimous / Conservative)', merged['jw_votes'] == 3),
]

thresh_results = []
for name, cond in thresholds:
    pred = np.where(cond, 'jaywalking', 'compliant')
    thresh_results.append({'Policy Threshold': name, **calc_metrics(merged['ground_truth'], pred)})

print(pd.DataFrame(thresh_results).to_string(index=False))

print('\n=== 3. DETAILED INSPECTION OF ALL 2/3 (SPLIT) CASES ===')
sub_2_3 = merged[merged['jw_votes'] == 2]
print(f'Total 2/3 split cases: {len(sub_2_3)}')
for idx, r in sub_2_3.iterrows():
    c = r['clip_name']
    gt = r['ground_truth']
    status = 'TP (True Jaywalker)' if gt == 'jaywalking' else 'FP (False Alarm)'
    print(f"- Clip: `{c}` | GT: `{gt}` | Status: **{status}** | Reason: {r['reason']}")
