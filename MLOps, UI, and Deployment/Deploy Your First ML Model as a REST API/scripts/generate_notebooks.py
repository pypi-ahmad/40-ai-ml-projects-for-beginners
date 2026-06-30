"""Generate zero-to-hero tutorial notebooks for Project #9."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NOTEBOOK_DIR = Path("notebooks")


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def bootstrap_cell():
    return code(
        "from pathlib import Path\n"
        "import sys\n"
        "\n"
        "project_root = Path.cwd()\n"
        "if not (project_root / 'src').exists():\n"
        "    project_root = project_root.parent\n"
        "if str(project_root) not in sys.path:\n"
        "    sys.path.insert(0, str(project_root))\n"
        "print('Project root:', project_root)"
    )


def section_template(title: str, motivation: str, real_world: str, mistakes: str) -> str:
    return f"""
## {title}

### Definition
{title} in this project means we design, validate, serve, and observe ML predictions through stable HTTP interfaces.

### Theory
APIs decouple producers (model service) and consumers (web app, batch jobs, partner systems). Typed contracts reduce ambiguity and production failures.

### Motivation
{motivation}

### Real-World Example
{real_world}

### Visual Explanation
Diagram/code cell below shows component flow and responsibilities.

### Code Explanation
Code cells in this section are structured as: setup → implementation → validation output.

### Best Practices
- Keep contracts explicit and versioned.
- Validate early at boundary.
- Log request IDs for traceability.
- Measure latency, error rate, and throughput.

### Common Mistakes
{mistakes}
"""


def build_notebook_01() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# Notebook 01 — API Fundamentals + REST Lifecycle\n\n"
            "This notebook explains API-first ML systems from zero and builds intuition before implementation."
        ),
        bootstrap_cell(),
        md(
            section_template(
                title="What Is an API?",
                motivation="Without APIs, every consumer would need direct model runtime access, creating coupling and security risk.",
                real_world="Ride-sharing ETA models are consumed by mobile app, pricing service, and support dashboards through API endpoints.",
                mistakes="- Treating API as only transport layer and forgetting validation/security.\n- Returning unstructured error text instead of typed error schemas.",
            )
        ),
        md(
            "## Restaurant Analogy\n\n"
            "- Client: customer placing order\n"
            "- Server: waiter carrying request/response\n"
            "- Kitchen: ML model inference engine\n"
            "- Menu item: endpoint contract\n"
            "- Receipt: HTTP status + JSON response"
        ),
        code(
            "import matplotlib.pyplot as plt\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(12, 3))\n"
            "ax.axis('off')\n"
            "labels = ['Client', 'REST API', 'Validation Layer', 'ML Model', 'Response']\n"
            "for i, lbl in enumerate(labels):\n"
            "    ax.text(i * 2.2, 0.5, lbl, ha='center', va='center',\n"
            "            bbox=dict(boxstyle='round,pad=0.4', fc='#d9edf7', ec='#31708f'))\n"
            "for i in range(len(labels)-1):\n"
            "    ax.annotate('', xy=(i*2.2+0.95, 0.5), xytext=(i*2.2+1.25, 0.5),\n"
            "                arrowprops=dict(arrowstyle='->', lw=2))\n"
            "plt.title('Request-Response Lifecycle')\n"
            "plt.show()"
        ),
        md(
            section_template(
                title="REST Fundamentals",
                motivation="REST gives predictable semantics for integration and long-term maintenance.",
                real_world="Fraud scoring APIs use GET for health/model metadata and POST for scoring payloads.",
                mistakes="- Using GET for side-effect operations.\n- Ignoring status codes and returning 200 for all failures.",
            )
        ),
        md(
            "### HTTP Methods Quick Map\n\n"
            "- `GET`: fetch state (health, metadata, metrics)\n"
            "- `POST`: create prediction/explanation results\n"
            "- `PUT`: full resource replacement\n"
            "- `PATCH`: partial updates\n"
            "- `DELETE`: remove resource"
        ),
        md(
            "### Status Codes\n\n"
            "- 200 OK: request succeeded\n"
            "- 201 Created: resource created\n"
            "- 400 Bad Request: malformed payload\n"
            "- 401 Unauthorized: auth required\n"
            "- 422 Unprocessable Entity: validation failed\n"
            "- 500 Internal Server Error: unexpected failure"
        ),
    ]
    return nb


def build_notebook_02() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Notebook 02 — FastAPI + Pydantic Deep Dive"),
        bootstrap_cell(),
        md(
            section_template(
                title="FastAPI Fundamentals",
                motivation="FastAPI gives type-driven validation and auto docs, reducing boilerplate in ML service development.",
                real_world="Internal recommendation microservices often standardize on FastAPI for speed + OpenAPI compatibility.",
                mistakes="- Skipping response models and returning arbitrary dicts.\n- Pushing business logic directly into route handlers.",
            )
        ),
        md(
            "## Framework Tradeoffs\n\n"
            "- Flask: flexible, minimal, but manual typing/docs.\n"
            "- FastAPI: strong typing, async-ready, automatic docs.\n"
            "- Django: batteries included; heavier for pure inference APIs.\n"
            "- Node.js APIs: high ecosystem breadth; type safety depends on stack choice."
        ),
        code(
            "from pydantic import BaseModel, Field, ValidationError\n"
            "\n"
            "class HousingRecord(BaseModel):\n"
            "    MedInc: float = Field(ge=0.0, le=30.0)\n"
            "    HouseAge: float = Field(ge=0.0, le=100.0)\n"
            "    AveRooms: float = Field(ge=0.1, le=100.0)\n"
            "    AveBedrms: float = Field(ge=0.1, le=20.0)\n"
            "    Population: float = Field(ge=1.0, le=100000.0)\n"
            "    AveOccup: float = Field(ge=0.1, le=100.0)\n"
            "    Latitude: float = Field(ge=32.0, le=43.0)\n"
            "    Longitude: float = Field(ge=-125.0, le=-113.0)\n"
            "\n"
            "valid = {\n"
            "    'MedInc': 8.3252, 'HouseAge': 41.0, 'AveRooms': 6.9841,\n"
            "    'AveBedrms': 1.0238, 'Population': 322.0, 'AveOccup': 2.5556,\n"
            "    'Latitude': 37.88, 'Longitude': -122.23\n"
            "}\n"
            "print('Valid model:', HousingRecord(**valid).model_dump())\n"
            "\n"
            "invalid = dict(valid)\n"
            "invalid['Longitude'] = -200.0\n"
            "try:\n"
            "    HousingRecord(**invalid)\n"
            "except ValidationError as exc:\n"
            "    print('Validation error snippet:')\n"
            "    print(exc.errors()[0])"
        ),
        md(
            "## Interactive Documentation\n\n"
            "FastAPI auto-generates:\n"
            "- Swagger UI at `/docs`\n"
            "- ReDoc at `/redoc`\n"
            "- OpenAPI JSON at `/openapi.json`\n\n"
            "Why it matters: typed contracts become discoverable and testable by other teams/tools."
        ),
    ]
    return nb


def build_notebook_03() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Notebook 03 — California Housing EDA + Baseline Modeling"),
        bootstrap_cell(),
        md(
            section_template(
                title="Dataset and Problem Definition",
                motivation="We need reproducible understanding of data before benchmarking many models.",
                real_world="Real estate pricing services use structured features to estimate median house values for decision support.",
                mistakes="- Data leakage from fitting preprocessing on all data.\n- Optimizing only one metric without business interpretation.",
            )
        ),
        code(
            "import pandas as pd\n"
            "import seaborn as sns\n"
            "import matplotlib.pyplot as plt\n"
            "from sklearn.linear_model import LinearRegression\n"
            "\n"
            "from src.data import load_california_housing, split_dataset\n"
            "from src.evaluation import compute_regression_metrics\n"
            "\n"
            "X, y = load_california_housing()\n"
            "print('Shape:', X.shape)\n"
            "display(X.head())\n"
            "display(y.head())\n"
            "split = split_dataset(X, y, random_state=42)\n"
            "print('Train/Val/Test:', len(split.X_train), len(split.X_val), len(split.X_test))"
        ),
        code(
            "plt.figure(figsize=(8, 5))\n"
            "sns.histplot(y, kde=True)\n"
            "plt.title('Target Distribution: MedHouseVal')\n"
            "plt.show()"
        ),
        md(
            "## Metric Theory\n\n"
            "- MAE: average absolute error, robust interpretation.\n"
            "- MSE: squared error, stronger penalty for large mistakes.\n"
            "- RMSE: MSE root, same unit as target.\n"
            "- R²: explained variance proportion.\n"
            "- MAPE: percentage error, scale independent."
        ),
        code(
            "model = LinearRegression()\n"
            "model.fit(split.X_train, split.y_train)\n"
            "pred = model.predict(split.X_val)\n"
            "metrics = compute_regression_metrics(split.y_val, pred)\n"
            "print(metrics)"
        ),
    ]
    return nb


def build_notebook_04() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Notebook 04 — Model Benchmarking: Manual + LazyPredict + FLAML + PyCaret"),
        bootstrap_cell(),
        md(
            section_template(
                title="Model Benchmarking",
                motivation="A single model baseline can hide significant quality/performance gains from alternatives.",
                real_world="Production teams compare multiple algorithms before locking serving contract and SLO budgets.",
                mistakes="- Benchmarking on test split repeatedly.\n- Ignoring runtime and deployment complexity in model selection.",
            )
        ),
        code(
            "from src.data import load_california_housing, split_dataset\n"
            "from src.training import train_and_rank_models\n"
            "from src.benchmarking import run_lazypredict_benchmark, run_flaml_benchmark, run_pycaret_benchmark\n"
            "\n"
            "X, y = load_california_housing()\n"
            "split = split_dataset(X, y, random_state=42)\n"
            "trained, ranking = train_and_rank_models(split.X_train, split.y_train, split.X_val, split.y_val)\n"
            "display(ranking.head(10))"
        ),
        code(
            "lazy = run_lazypredict_benchmark(split.X_train, split.X_val, split.y_train, split.y_val)\n"
            "flaml = run_flaml_benchmark(split.X_train, split.X_val, split.y_train, split.y_val, time_budget_seconds=30)\n"
            "pyc = run_pycaret_benchmark(split.X_train, split.X_val, split.y_train, split.y_val)\n"
            "\n"
            "print('LazyPredict:', lazy.status, lazy.notes)\n"
            "print('FLAML:', flaml.status, flaml.notes)\n"
            "print('PyCaret:', pyc.status, pyc.notes)\n"
            "if not lazy.table.empty:\n"
            "    display(lazy.table.head())\n"
            "if not flaml.table.empty:\n"
            "    display(flaml.table.head())\n"
            "if not pyc.table.empty:\n"
            "    display(pyc.table.head())"
        ),
        md(
            "## Tradeoff Discussion\n\n"
            "- LazyPredict: fastest broad scan, low control.\n"
            "- FLAML: efficient AutoML with time budget control.\n"
            "- PyCaret: high-level experimentation workflow, heavier dependency/runtime footprint."
        ),
    ]
    return nb


def build_notebook_05() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Notebook 05 — Serialization, Serving Architecture, and Explainability"),
        bootstrap_cell(),
        md(
            section_template(
                title="Model Serialization",
                motivation="Training and serving often run in different processes/machines; model artifact portability is required.",
                real_world="Daily retraining jobs publish signed artifacts consumed by online inference services.",
                mistakes="- Loading untrusted pickle artifacts.\n- Missing metadata (feature order/version/metrics) alongside model binary.",
            )
        ),
        md(
            "## Pickle vs Joblib vs ONNX\n\n"
            "- Pickle: standard Python serialization, broad but unsafe with untrusted files.\n"
            "- Joblib: efficient for NumPy/scikit-learn artifacts, common ML ops default.\n"
            "- ONNX: cross-runtime portability, extra conversion complexity."
        ),
        code(
            "import json\n"
            "from pathlib import Path\n"
            "\n"
            "from src.data import load_california_housing, split_dataset\n"
            "from src.serialization import load_metadata\n"
            "from src.training import train_best_model, build_metadata\n"
            "\n"
            "X, y = load_california_housing()\n"
            "split = split_dataset(X, y, random_state=42)\n"
            "best, ranking = train_best_model(split.X_train, split.y_train, split.X_val, split.y_val, split.X_test, split.y_test)\n"
            "meta = build_metadata(best, ranking, len(split.X_train), len(split.X_val), len(split.X_test))\n"
            "print(json.dumps({k: meta[k] for k in ['model_name', 'rmse', 'r2', 'mape']}, indent=2))"
        ),
        code(
            "import matplotlib.pyplot as plt\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(12, 3))\n"
            "ax.axis('off')\n"
            "blocks = ['Client', 'REST API', 'Validation', 'Model Artifact', 'Prediction']\n"
            "for i, b in enumerate(blocks):\n"
            "    ax.text(i * 2.3, 0.5, b, ha='center', va='center',\n"
            "            bbox=dict(boxstyle='round,pad=0.3', fc='#f7f7d9', ec='#8a8a00'))\n"
            "for i in range(len(blocks)-1):\n"
            "    ax.annotate('', xy=(i*2.3+1.0, 0.5), xytext=(i*2.3+1.3, 0.5), arrowprops=dict(arrowstyle='->'))\n"
            "plt.title('Production Inference Architecture')\n"
            "plt.show()"
        ),
        md(
            "## Explainable AI with SHAP\n\n"
            "Use SHAP to expose feature contribution per prediction.\n"
            "API endpoint `/explain` should return base value + per-feature contributions."
        ),
    ]
    return nb


def build_notebook_06() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Notebook 06 — API Testing, Error Handling, Monitoring, and Performance"),
        bootstrap_cell(),
        md(
            section_template(
                title="Testing and Production Engineering",
                motivation="Model quality without reliable API behavior is not production-ready.",
                real_world="ML fraud APIs require strict SLAs, predictable errors, and auditable request tracking.",
                mistakes="- Manual testing only.\n- Missing negative tests for malformed payloads and auth failures.",
            )
        ),
        code(
            "import httpx\n"
            "\n"
            "payload = {\n"
            "    'MedInc': 8.3252, 'HouseAge': 41.0, 'AveRooms': 6.9841,\n"
            "    'AveBedrms': 1.0238, 'Population': 322.0, 'AveOccup': 2.5556,\n"
            "    'Latitude': 37.88, 'Longitude': -122.23\n"
            "}\n"
            "\n"
            "try:\n"
            "    r = httpx.get('http://127.0.0.1:8000/health', timeout=3.0)\n"
            "    print('Health status:', r.status_code, r.json())\n"
            "    rp = httpx.post('http://127.0.0.1:8000/predict', json=payload, timeout=5.0)\n"
            "    print('Predict status:', rp.status_code)\n"
            "    print('Predict body keys:', list(rp.json().keys()))\n"
            "except Exception as exc:\n"
            "    print('API not running locally; skip live call test:', exc)"
        ),
        code(
            "import pandas as pd\n"
            "from pathlib import Path\n"
            "\n"
            "perf_path = Path('artifacts/performance/api_performance_summary.json')\n"
            "if perf_path.exists():\n"
            "    display(pd.read_json(perf_path, typ='series'))\n"
            "else:\n"
            "    print('Run scripts/benchmark_api.py after API is up to generate performance report.')"
        ),
        md(
            "## Security Basics\n\n"
            "- Strict Pydantic validation at boundary.\n"
            "- Optional API key for protected inference endpoints.\n"
            "- Rate limiting to reduce abuse burst risk.\n"
            "- Request ID correlation for incident debugging."
        ),
    ]
    return nb


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    notebooks = {
        "01_api_fundamentals_rest.ipynb": build_notebook_01(),
        "02_fastapi_pydantic.ipynb": build_notebook_02(),
        "03_dataset_eda_modeling.ipynb": build_notebook_03(),
        "04_benchmarking_and_selection.ipynb": build_notebook_04(),
        "05_serialization_serving_xai.ipynb": build_notebook_05(),
        "06_testing_monitoring_performance.ipynb": build_notebook_06(),
    }

    for name, notebook in notebooks.items():
        path = NOTEBOOK_DIR / name
        with path.open("w", encoding="utf-8") as f:
            nbf.write(notebook, f)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
