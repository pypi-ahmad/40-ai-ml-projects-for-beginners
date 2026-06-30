"""
fix_notebooks.py
----------------
Repair notebook/API drift for notebooks 02-07.

What this script fixes:
1) FeatureSelector stage-A/B calls that forgot to pass X.
2) Legacy pipeline kwargs in notebook 06 (unsupported in current API).
3) Broken SHAP analysis cell in notebook 07 (collapsed into one line).

Usage:
    python src/fix_notebooks.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

NOTEBOOKS = [
    "02_real_dataset_exploration.ipynb",
    "03_feature_selection_funnel.ipynb",
    "04_benchmarking.ipynb",
    "05_advanced_visualizations.ipynb",
    "06_pipeline_shap_inference.ipynb",
    "07_error_analysis.ipynb",
]


PIPELINE_CELL_06 = """sel = FeatureSelector()
selected_features = sel.pipeline(
    X_train,
    y_train,
    X_val=X_test,
    y_val=y_test,
    var_threshold=0.0,
    corr_threshold=0.99,
    rfe_feat=120,
    l1_C=1.0,
    mi_k=120,
    shap_k=120,
    verbose=True,
)
X_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
print(f"\\nFinal: {X_train.shape[1]} features -> {len(selected_features)} features")
"""


SHAP_CELL_07 = """explainer = shap.TreeExplainer(model, X_sel[:100], feature_perturbation=\"interventional\")
shap_values = explainer.shap_values(X_sel[:200], check_additivity=False)

if isinstance(shap_values, list):
    sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
elif getattr(shap_values, "ndim", 2) == 3:
    sv = shap_values[:, :, -1]
else:
    sv = shap_values

# SHAP summary
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(sv, X_sel[:200], show=False)
plt.tight_layout()
plt.show()

# Dependence plots for top features
mean_shap = np.abs(sv).mean(axis=0)
top_feature_idx = np.argsort(mean_shap)[-3:][::-1]
for idx in top_feature_idx:
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(int(idx), sv, X_sel[:200], show=False)
    plt.tight_layout()
    plt.show()

# Error SHAP analysis
error_mask = errors[: len(sv)] if len(errors) >= len(sv) else errors
error_indices_local = np.where(error_mask)[0]

if len(error_indices_local) > 0:
    correct_indices_local = np.where(~error_mask)[0]
    shap_errors = sv[error_indices_local]
    shap_correct = sv[correct_indices_local] if len(correct_indices_local) > 0 else np.empty((0, sv.shape[1]))

    shap_error_mean = np.abs(shap_errors).mean(axis=0)
    shap_correct_mean = np.abs(shap_correct).mean(axis=0) if shap_correct.size else np.zeros_like(shap_error_mean)

    feature_names = X_sel.columns[: sv.shape[1]]
    diff_df = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_abs_error": shap_error_mean,
            "shap_abs_correct": shap_correct_mean,
        }
    )
    diff_df["diff"] = diff_df["shap_abs_error"] - diff_df["shap_abs_correct"]
    diff_df = diff_df.sort_values("diff", ascending=False)
    print("Features with highest SHAP difference (errors vs correct):")
    display(diff_df.head(10))
"""


def _fix_stage_calls(src: str) -> Tuple[str, bool]:
    changed = False

    # Add X=X_train to stage A when missing
    pattern_a = re.compile(
        r"(\b(?:sel|selector)\.variance_threshold\()\s*threshold\s*=\s*([^,\)]+)\s*\)",
    )

    def repl_a(match: re.Match) -> str:
        nonlocal changed
        changed = True
        return f"{match.group(1)}threshold={match.group(2)}, X=X_train)"

    src = pattern_a.sub(repl_a, src)

    # Add X=X_train to stage B when missing
    pattern_b = re.compile(
        r"(\b(?:sel|selector)\.correlation_filter\()\s*threshold\s*=\s*([^,\)]+)\s*\)",
    )

    def repl_b(match: re.Match) -> str:
        nonlocal changed
        changed = True
        return f"{match.group(1)}threshold={match.group(2)}, X=X_train)"

    src = pattern_b.sub(repl_b, src)

    return src, changed


def _fix_notebook_specific(nb_name: str, src: str) -> Tuple[str, bool]:
    if nb_name == "06_pipeline_shap_inference.ipynb":
        if "importance_threshold=" in src and ".pipeline(" in src:
            return PIPELINE_CELL_06, True

    if nb_name == "07_error_analysis.ipynb":
        if "shap_values# Dependence plots" in src:
            return SHAP_CELL_07, True

    return src, False


def fix_notebook(nb_path: Path) -> bool:
    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        src = "".join(cell.get("source", []))
        new_src, c1 = _fix_notebook_specific(nb_path.name, src)
        new_src, c2 = _fix_stage_calls(new_src)

        if c1 or c2:
            cell["source"] = [line + "\n" for line in new_src.rstrip("\n").split("\n")]
            changed = True

    if changed:
        with nb_path.open("w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
        print(f"✓ {nb_path.name} — fixed")
    else:
        print(f"  {nb_path.name} — no changes needed")

    return changed


def main() -> None:
    print(f"Notebook directory: {NOTEBOOKS_DIR}")
    for name in NOTEBOOKS:
        path = NOTEBOOKS_DIR / name
        if not path.exists():
            print(f"  {name} not found, skipping")
            continue
        fix_notebook(path)


if __name__ == "__main__":
    main()
