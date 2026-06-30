"""Generate zero-to-hero tutorial notebooks for ML packaging project."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


COMMON_SETUP = """
from pathlib import Path
import os

CWD = Path.cwd()
ROOT = CWD if (CWD / "pyproject.toml").exists() else CWD.parent
os.chdir(ROOT)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path("outputs/figures").mkdir(parents=True, exist_ok=True)
Path("outputs/benchmarks").mkdir(parents=True, exist_ok=True)
"""


def build_notebook_01() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            """
# 01 - Model Packaging Foundations (Zero to Hero)

This notebook starts from first principles and teaches **what model packaging means** in production ML.

## Definition
Model packaging is process of turning trained model into reusable software artifact with versioning, metadata, and stable interfaces.

## Theory
Raw training code alone is not enough for production. Teams need deterministic artifacts, dependency boundaries, and predictable API contracts.

## Motivation
Without packaging, deployment becomes manual and fragile. With packaging, one trained model can be reused by API, CLI, batch jobs, and tests.

## Real-World Example
Fraud model trained by one team can be versioned and consumed by:
- Realtime API for card authorizations
- Batch risk scoring pipeline
- Internal analyst notebook tools

## Best Practices
- Keep model artifacts immutable
- Attach checksums + metadata
- Track model versions in registry
- Separate training and serving concerns

## Common Mistakes
- Saving only one `.pkl` with no metadata
- No input validation layer
- No rollback strategy
- No reproducible dependency lock
"""
        ),
        code(COMMON_SETUP),
        md(
            """
## Lifecycle Theory and Visual Explanation

### Definition
Lifecycle means complete flow from raw data to production inference.

### Theory
Flow stages:
1. Raw Data
2. Training
3. Model Artifact
4. Packaging
5. Serving
6. Production

### Motivation
Each stage introduces risk. Packaging stage is control point where model becomes governed software.
"""
        ),
        code(
            """
import matplotlib.pyplot as plt

steps = ["Raw Data", "Training", "Model Artifact", "Packaging", "Serving", "Production"]
x = list(range(len(steps)))

plt.figure(figsize=(12, 2.8))
for idx, label in enumerate(steps):
    plt.text(
        idx, 0.5, label, ha="center", va="center",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f1f3f5", "edgecolor": "#495057"},
    )
for idx in range(len(steps) - 1):
    plt.annotate("", xy=(idx + 0.7, 0.5), xytext=(idx + 0.3, 0.5), arrowprops={"arrowstyle": "->", "lw": 1.8})
plt.xlim(-0.5, len(steps) - 0.5)
plt.ylim(0, 1)
plt.axis("off")
plt.title("ML Packaging Lifecycle")
out = Path("outputs/figures/notebook01_lifecycle.png")
plt.savefig(out, dpi=220, bbox_inches="tight")
plt.show()
print(f"Saved: {out}")
"""
        ),
        md(
            """
## Packaging Architecture (Client -> API -> Wrapper -> Artifact -> Prediction)

### Theory
Wrapper layer decouples model file format from serving interface.

### Code Explanation
The code below draws reference architecture used in this project.
"""
        ),
        code(
            """
import matplotlib.pyplot as plt

blocks = ["Client", "API Layer", "Wrapper Layer", "Model Artifact", "Prediction"]
xs = [0.08, 0.28, 0.50, 0.72, 0.90]

fig, ax = plt.subplots(figsize=(10, 3))
ax.axis("off")
for label, x in zip(blocks, xs):
    ax.text(
        x, 0.5, label, ha="center", va="center",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#e7f5ff", "edgecolor": "#1c7ed6"},
        transform=ax.transAxes,
    )
for start, end in zip(xs[:-1], xs[1:]):
    ax.annotate("", xy=(end - 0.06, 0.5), xytext=(start + 0.06, 0.5), arrowprops={"arrowstyle": "->", "lw": 1.6}, xycoords=ax.transAxes)
ax.set_title("Production Model Packaging Architecture")
out = Path("outputs/figures/notebook01_architecture.png")
fig.savefig(out, dpi=220, bbox_inches="tight")
plt.show()
print(f"Saved: {out}")
"""
        ),
    ]
    return nb


def build_notebook_02() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            """
# 02 - Iris Dataset Understanding and EDA

## Definition
Exploratory Data Analysis (EDA) is process of inspecting structure, distributions, and relationships before modeling.

## Theory
Good packaging starts with clear understanding of feature ranges and class balance so validation boundaries are realistic.

## Motivation
Validation rules in API/CLI should reflect real feature ranges from training data.

## Real-World Example
If production payloads drift far beyond training ranges, predictions become unstable and risky.
"""
        ),
        code(COMMON_SETUP),
        code(
            """
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
df.head()
"""
        ),
        md(
            """
## Feature Analysis and Class Analysis

### Code Explanation
We inspect shape, nulls, descriptive statistics, and class distribution.
"""
        ),
        code(
            """
print("Shape:", df.shape)
print("\\nNull values:\\n", df.isnull().sum())
print("\\nClass counts:\\n", df["species"].value_counts())
df.describe().T
"""
        ),
        md(
            """
## Visual Explanation: Histograms, Boxplots, Pairplot, Correlation

### Best Practices
- Use multiple plot types, not one chart only
- Save visuals for reproducible reporting
- Interpret plots in context of business/API validation

### Common Mistakes
- Trusting one metric or one plot
- Ignoring outliers before setting validation thresholds
"""
        ),
        code(
            """
num_cols = [c for c in df.columns if c not in {"target", "species"}]

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axes.flat, num_cols):
    sns.histplot(df[col], kde=True, ax=ax, color="#0b7285")
    ax.set_title(f"Histogram: {col}")
fig.tight_layout()
hist_path = Path("outputs/figures/notebook02_histograms.png")
fig.savefig(hist_path, dpi=220, bbox_inches="tight")
plt.show()
print(f"Saved: {hist_path}")
"""
        ),
        code(
            """
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axes.flat, num_cols):
    sns.boxplot(data=df, x="species", y=col, ax=ax)
    ax.set_title(f"Boxplot: {col}")
fig.tight_layout()
box_path = Path("outputs/figures/notebook02_boxplots.png")
fig.savefig(box_path, dpi=220, bbox_inches="tight")
plt.show()
print(f"Saved: {box_path}")
"""
        ),
        code(
            """
pair = sns.pairplot(df[num_cols + ["species"]], hue="species", corner=True, plot_kws={"s": 28, "alpha": 0.8})
pair.fig.suptitle("Pair Plot by Species", y=1.02)
pair_path = Path("outputs/figures/notebook02_pairplot.png")
pair.fig.savefig(pair_path, dpi=220, bbox_inches="tight")
plt.show()
print(f"Saved: {pair_path}")
"""
        ),
        code(
            """
corr = df[num_cols].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f")
plt.title("Feature Correlation Matrix")
corr_path = Path("outputs/figures/notebook02_correlation.png")
plt.savefig(corr_path, dpi=220, bbox_inches="tight")
plt.show()
print(f"Saved: {corr_path}")
"""
        ),
        md(
            """
## Findings Summary
- Dataset is balanced across 3 classes
- Petal features show strongest separation
- Correlated features matter for model and explainability interpretation
"""
        ),
    ]
    return nb


def build_notebook_03() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            """
# 03 - Benchmarking Models and Selecting Production Candidate

## Definition
Benchmarking compares multiple models under same split and metrics.

## Theory
Production selection should optimize tradeoff: quality + latency + maintainability.

## Motivation
Do not choose model by intuition. Use reproducible ranking with clear criteria.
"""
        ),
        code(COMMON_SETUP),
        code(
            """
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

bench_csv = Path("outputs/benchmarks/model_benchmark.csv")
automl_json = Path("outputs/benchmarks/automl_benchmark.json")

if not bench_csv.exists() or not automl_json.exists():
    subprocess.run([sys.executable, "scripts/train_model.py"], check=True)

benchmark_df = pd.read_csv(bench_csv)
benchmark_df.sort_values(["f1_macro", "accuracy"], ascending=False)
"""
        ),
        md(
            """
## Required Model Families
This table includes:
- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- XGBoost
- LightGBM
- CatBoost
- SVM
- KNN

Optional libraries load when installed.
"""
        ),
        code(
            """
ranked = benchmark_df[benchmark_df["status"] == "ok"].sort_values(["f1_macro", "accuracy"], ascending=False)
ranked[["model_name", "accuracy", "precision_macro", "recall_macro", "f1_macro", "predict_time_ms"]]
"""
        ),
        code(
            """
automl_df = pd.read_json("outputs/benchmarks/automl_benchmark.json")
automl_df
"""
        ),
        md(
            """
## LazyPredict vs FLAML vs PyCaret

### Why Each Exists
- **LazyPredict**: rapid baseline sweep
- **FLAML**: efficient AutoML search with time budget
- **PyCaret**: low-code experiment management and model comparison

### Strengths
- LazyPredict: speed and simplicity
- FLAML: strong cost-performance efficiency
- PyCaret: rich experiment abstraction

### Weaknesses
- LazyPredict: less control and deeper tuning
- FLAML: less pedagogical transparency than manual modeling
- PyCaret: heavier dependency stack

### Tradeoff
Use all three for teaching breadth, then pick production path based on control vs speed.
"""
        ),
        code(
            """
bench_path = Path("outputs/benchmarks/model_benchmark_from_notebook03.csv")
ranked.to_csv(bench_path, index=False)
print(f"Saved: {bench_path}")

automl_path = Path("outputs/benchmarks/automl_benchmark_from_notebook03.json")
automl_path.write_text(automl_df.to_json(orient="records", indent=2), encoding="utf-8")
print(f"Saved: {automl_path}")
"""
        ),
    ]
    return nb


def build_notebook_04() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            """
# 04 - Evaluation Metrics and Diagnostic Visualizations

## Definition
Evaluation quantifies model behavior beyond single accuracy number.

## Theory
Multi-class classification needs per-class and aggregate views.

## Motivation
Packaging decisions (version promotion, rollback, monitoring thresholds) need trustworthy metrics.
"""
        ),
        code(COMMON_SETUP),
        code(
            """
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)
"""
        ),
        code(
            """
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
    "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
    "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    "roc_auc_ovr": roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"),
}
metrics
"""
        ),
        md(
            """
## Metric Explanation
- **Accuracy**: overall correctness
- **Precision**: predicted class purity
- **Recall**: how much true class captured
- **F1**: precision-recall balance
- **ROC AUC (OvR)**: ranking quality per class
"""
        ),
        code(
            """
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=iris.target_names, cmap="Blues", ax=ax)
plt.title("Confusion Matrix")
cm_path = Path("outputs/figures/notebook04_confusion_matrix.png")
plt.savefig(cm_path, dpi=220, bbox_inches="tight")
plt.show()
print(f"Saved: {cm_path}")
"""
        ),
        code(
            """
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for idx, name in enumerate(iris.target_names):
    RocCurveDisplay.from_predictions(y_test_bin[:, idx], y_prob[:, idx], name=name, ax=axes[0])
axes[0].set_title("One-vs-Rest ROC Curves")

for idx, name in enumerate(iris.target_names):
    PrecisionRecallDisplay.from_predictions(y_test_bin[:, idx], y_prob[:, idx], name=name, ax=axes[1])
axes[1].set_title("One-vs-Rest Precision-Recall Curves")

fig.tight_layout()
curve_path = Path("outputs/figures/notebook04_roc_pr_curves.png")
fig.savefig(curve_path, dpi=220, bbox_inches="tight")
plt.show()
print(f"Saved: {curve_path}")
"""
        ),
    ]
    return nb


def build_notebook_05() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            """
# 05 - Serialization Deep Dive (Pickle, Joblib, ONNX, TorchScript)

## Definition
Serialization converts in-memory model into portable byte artifact.

## Theory
Different formats optimize different goals: portability, speed, security, ecosystem support.

## Motivation
Packaging engineers must choose format intentionally, not by habit.
"""
        ),
        code(COMMON_SETUP),
        md(
            """
## Security Considerations
- Pickle/Joblib can execute arbitrary code when loading untrusted files
- Always verify source and checksum
- Use manifest + trusted digest allow-list in production
"""
        ),
        code(
            """
from pathlib import Path
import json

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from ml_package.serialization_benchmark import benchmark_serialization, write_serialization_benchmark
from ml_package.model_loader import ModelLoader

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = RandomForestClassifier(n_estimators=250, random_state=42)
model.fit(X_train, y_train)

rows = benchmark_serialization(
    model,
    artifact_stem="notebook05_iris_model",
    output_dir="models",
)
out_path = write_serialization_benchmark(rows, "outputs/benchmarks/notebook05_serialization_benchmark.json")
print(f"Saved: {out_path}")
rows
"""
        ),
        code(
            """
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame(rows)
display(df)

ok = df[df["status"] == "ok"]
if not ok.empty:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.barplot(data=ok, x="format", y="save_time_ms", ax=axes[0], color="#0b7285")
    sns.barplot(data=ok, x="format", y="load_time_ms", ax=axes[0], color="#74c0fc", alpha=0.65)
    axes[0].set_title("Serialization Save/Load Time")
    axes[0].set_ylabel("Milliseconds")

    sns.barplot(data=ok, x="format", y=(ok["size_bytes"] / 1024), ax=axes[1], color="#228be6")
    axes[1].set_title("Artifact Size")
    axes[1].set_ylabel("KB")

    fig.tight_layout()
    fig_path = Path("outputs/figures/notebook05_serialization_compare.png")
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fig_path}")
"""
        ),
        md(
            """
## Tradeoff Summary
- **Pickle**: simple, broad Python compatibility, higher security risk if untrusted
- **Joblib**: better for numpy-heavy objects, same trust model as pickle
- **ONNX**: portable runtime, better cross-language deployment
- **TorchScript**: strong for PyTorch workflows, not primary for sklearn pipeline
"""
        ),
    ]
    return nb


def build_notebook_06() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            """
# 06 - Reusable Wrapper Layer, FastAPI, and Pydantic Validation

## Definition
Wrapper layer is software boundary that standardizes loading, validation, prediction, and metadata.

## Theory
Model artifact should never be called directly by external client. Wrapper controls input contract and error handling.

## Motivation
Without wrapper/API contract, model serving logic gets duplicated across services.
"""
        ),
        code(COMMON_SETUP),
        code(
            """
from pathlib import Path
import numpy as np

from ml_package import ModelLoader, PredictionEngine
from ml_package.validation import IrisValidator

model_path = Path("models/iris_model.pkl")
engine = PredictionEngine(ModelLoader(model_path, verify_integrity=True).load())
validator = IrisValidator()
"""
        ),
        md(
            """
## Validation Layer Demonstration

### Valid request
Should pass with no errors.

### Invalid request
Should return detailed field-level feedback.
"""
        ),
        code(
            """
valid_errors = validator.validate_request(5.1, 3.5, 1.4, 0.2)
invalid_errors = validator.validate_request(-1.0, 99.0, 0.0, -3.0)
print("Valid errors:", valid_errors)
print("Invalid errors:", invalid_errors)
"""
        ),
        code(
            """
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
pred = engine.predict(sample)
pred
"""
        ),
        md(
            """
## FastAPI Endpoint Walkthrough
- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /predict-batch`
- `GET /metrics`
- `GET /metrics/prometheus`
- `POST /explain` (`mode=local|global`)
"""
        ),
        code(
            """
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

health = client.get("/health")
print("Health:", health.status_code, health.json())

predict_payload = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}
pred_resp = client.post("/predict", json=predict_payload)
print("Predict:", pred_resp.status_code, pred_resp.json())

metrics_resp = client.get("/metrics/prometheus")
print("Prometheus metrics snippet:\\n", "\\n".join(metrics_resp.text.splitlines()[:6]))
"""
        ),
    ]
    return nb


def build_notebook_07() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            """
# 07 - Versioning, Testing Strategy, and CLI Operations

## Definition
Model versioning tracks lineage of artifacts and enables safe promotion/rollback.

## Theory
Registry metadata (metrics, hashes, parent version) provides auditability.

## Motivation
When production quality degrades, rollback must be explicit and fast.
"""
        ),
        code(COMMON_SETUP),
        code(
            """
import json
from pathlib import Path

registry_path = Path("models/registry.json")
registry = json.loads(registry_path.read_text(encoding="utf-8"))
registry
"""
        ),
        md(
            """
## Version Story in This Project
- `v1`: LogisticRegression baseline
- `v2`: benchmark-selected improved model

### Best Practice
Always keep parent-child linkage for traceability.
"""
        ),
        code(
            """
from ml_package.versioning import VersionRegistry

tmp_registry = Path("outputs/benchmarks/notebook07_registry.json")
if tmp_registry.exists():
    tmp_registry.unlink()

vr = VersionRegistry(str(tmp_registry))
vr.register("v1", "models/iris_model_v1.pkl", metrics={"f1_macro": 0.96}, description="baseline", allow_overwrite=True)
vr.activate("v1")
vr.register("v2", "models/iris_model_v2.pkl", metrics={"f1_macro": 1.00}, description="improved", parent_version="v1", allow_overwrite=True)
vr.activate("v2")
vr.rollback_to("v1")
vr.list_versions()
"""
        ),
        md(
            """
## Testing Framework

### Unit Tests
Validation, loader, prediction engine, version registry.

### Integration Tests
FastAPI routes through ASGI + HTTP client.

### API Live Tests
`requests` tests included with sandbox-aware skip when sockets are blocked.
"""
        ),
        code(
            """
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "tests/test_versioning.py", "tests/test_api.py"],
    capture_output=True,
    text=True,
    check=False,
)
print(result.stdout)
print("Exit code:", result.returncode)
"""
        ),
        md(
            """
## CLI Interface Demonstration
CLI is useful for batch scoring and automation where full API server is unnecessary.
"""
        ),
        code(
            """
import subprocess
import sys

cmd = [
    sys.executable,
    "-m",
    "ml_package.cli.predict",
    "--model-path",
    "models/iris_model.pkl",
    "predict",
    "--sepal-length",
    "5.1",
    "--sepal-width",
    "3.5",
    "--petal-length",
    "1.4",
    "--petal-width",
    "0.2",
]
out = subprocess.run(cmd, capture_output=True, text=True, check=False)
print(out.stdout)
print("CLI exit code:", out.returncode)
"""
        ),
    ]
    return nb


def build_notebook_08() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            """
# 08 - Monitoring, Explainability, Security, and Deployment Preparation

## Definition
Production readiness includes observability, explainability, and deployment controls.

## Theory
Good packaged model is not only accurate; it is measurable, explainable, and recoverable.

## Motivation
Operations teams need metrics, logs, and secure artifact handling to trust ML services.
"""
        ),
        code(COMMON_SETUP),
        code(
            """
from ml_package.logging_config import PredictionLogger

logger = PredictionLogger("notebook08.predictions")
logger.log_prediction(
    features={"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    prediction=0,
    confidence=0.98,
    latency_ms=2.45,
    model_version="v2",
)
logger.log_error([None, None, None, None], "validation_failed")
logger.get_stats()
"""
        ),
        md(
            """
## Explainable AI with SHAP

### Global explanation
Feature importance across reference set.

### Local explanation
Contribution for one sample prediction.
"""
        ),
        code(
            """
import numpy as np
from ml_package.model_loader import ModelLoader
from ml_package.explainability import ModelExplainer

model = ModelLoader("models/iris_model.pkl", verify_integrity=True).load()
background = np.load("models/background_data.npy")
explainer = ModelExplainer(model, background)

local = explainer.explain_single(np.array([[5.1, 3.5, 1.4, 0.2]], dtype=float))
global_imp = explainer.get_global_importance(background)

print("Local keys:", list(local.keys()))
print("Global keys:", list(global_imp.keys()))
"""
        ),
        md(
            """
## Security Considerations Checklist
- Unsafe deserialization risk for pickle/joblib
- Require checksum manifest in production
- Allow-list trusted digests
- Validate all API payloads with Pydantic
- Avoid direct artifact loading from untrusted sources
"""
        ),
        md(
            """
## Deployment Preparation

### Docker
Containerize API with immutable dependencies.

### Make Targets
Standardized commands for train/test/api/notebooks.

### Monitoring Endpoints
- JSON metrics: `/metrics`
- Prometheus metrics: `/metrics/prometheus`
"""
        ),
    ]
    return nb


def main() -> None:
    notebook_dir = Path("notebooks")
    notebook_dir.mkdir(parents=True, exist_ok=True)

    notebooks = [
        ("01_model_packaging_foundations.ipynb", build_notebook_01()),
        ("02_iris_eda_and_understanding.ipynb", build_notebook_02()),
        ("03_model_benchmarking_and_selection.ipynb", build_notebook_03()),
        ("04_evaluation_metrics_and_diagnostics.ipynb", build_notebook_04()),
        ("05_serialization_deep_dive.ipynb", build_notebook_05()),
        ("06_wrapper_api_validation.ipynb", build_notebook_06()),
        ("07_versioning_testing_and_cli.ipynb", build_notebook_07()),
        ("08_monitoring_explainability_and_deployment.ipynb", build_notebook_08()),
    ]

    for name, notebook in notebooks:
        path = notebook_dir / name
        nbf.write(notebook, path)
        print(f"Generated: {path}")


if __name__ == "__main__":
    main()
