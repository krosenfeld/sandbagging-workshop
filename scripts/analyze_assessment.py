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


def compute_metrics(y_true, y_pred):
    """Compute classification metrics and return as dict."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    n_samples = len(y_true)
    n_correct = int((y_true == y_pred).sum())
    baseline_rate = 0.5
    binom_result = binomtest(n_correct, n_samples, baseline_rate, alternative="greater")

    return {
        "n_samples": n_samples,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cm": cm,
        "n_correct": n_correct,
        "p_value": binom_result.pvalue,
    }


def print_metrics(metrics, title="Classification Metrics"):
    """Print metrics in a formatted way."""
    print(title)
    print(f"  Samples:   {metrics['n_samples']}")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1 Score:  {metrics['f1']:.3f}")
    print()
    print("  Confusion Matrix:")
    cm = metrics["cm"]
    print(f"    TN={cm[0, 0]:3d}  FP={cm[0, 1]:3d}")
    print(f"    FN={cm[1, 0]:3d}  TP={cm[1, 1]:3d}")
    print()
    print("  Statistical Test (Binomial):")
    print(f"    H0: accuracy <= 50.0% (random chance)")
    print(
        f"    Observed: {metrics['n_correct']}/{metrics['n_samples']} correct ({metrics['accuracy']:.1%})"
    )
    print(f"    p-value:  {metrics['p_value']:.4f}")
    result_str = (
        "Reject H0 (p < 0.05)" if metrics["p_value"] < 0.05 else "Fail to reject H0"
    )
    print(f"    Result:   {result_str}")


# Load data
evals = evals_df(paths.logs)
samples = samples_df(paths.logs)

# Extract predictions from choices column (stored as string like '["no"]')
samples["prediction"] = samples["choices"].apply(lambda x: ast.literal_eval(x)[0])

# Convert to binary labels (1 for 'yes', 0 for 'no')
samples["y_true"] = (samples["target"] == "yes").astype(int)
samples["y_pred"] = (samples["prediction"] == "yes").astype(int)

# Merge with evals to get model info
samples_with_model = samples.merge(evals[["eval_id", "model"]], on="eval_id")

# Report metrics by model
print("=" * 60)
print("METRICS BY MODEL")
print("=" * 60)

for model in samples_with_model["model"].unique():
    model_samples = samples_with_model[samples_with_model["model"] == model]
    y_true = model_samples["y_true"].values
    y_pred = model_samples["y_pred"].values

    metrics = compute_metrics(y_true, y_pred)
    print()
    print_metrics(metrics, title=f"Model: {model}")
    print()
    print("-" * 60)
