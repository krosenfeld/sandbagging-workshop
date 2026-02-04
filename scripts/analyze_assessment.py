import ast

from inspect_ai.analysis import evals_df, samples_df
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from sandbagging_workshop import paths

evals = evals_df(paths.logs)
samples = samples_df(paths.logs)

# Extract predictions from choices column (stored as string like '["no"]')
samples["prediction"] = samples["choices"].apply(lambda x: ast.literal_eval(x)[0])

# Convert to binary labels (1 for 'yes', 0 for 'no')
y_true = (samples["target"] == "yes").astype(int)
y_pred = (samples["prediction"] == "yes").astype(int)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print(f"Number of samples: {len(samples)}")
print()
print("Classification Metrics:")
print(f"  Accuracy:  {accuracy:.3f}")
print(f"  Precision: {precision:.3f}")
print(f"  Recall:    {recall:.3f}")
print(f"  F1 Score:  {f1:.3f}")
print()
print("Confusion Matrix:")
print(f"  TN={cm[0, 0]:3d}  FP={cm[0, 1]:3d}")
print(f"  FN={cm[1, 0]:3d}  TP={cm[1, 1]:3d}")

# Statistical test: is accuracy significantly better than random?
n_samples = len(y_true)
n_correct = int((y_true == y_pred).sum())
baseline_rate = 0.5  # random guessing

result = binomtest(n_correct, n_samples, baseline_rate, alternative="greater")
print()
print("Statistical Test (Binomial):")
print(f"  H0: accuracy <= {baseline_rate:.1%} (random chance)")
print(f"  Observed: {n_correct}/{n_samples} correct ({accuracy:.1%})")
print(f"  p-value:  {result.pvalue:.4f}")
print(
    f"  Result:   {'Reject H0 (p < 0.05)' if result.pvalue < 0.05 else 'Fail to reject H0'}"
)
