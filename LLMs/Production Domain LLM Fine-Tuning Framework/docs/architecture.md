# Architecture Diagram

```mermaid
flowchart TD
  C[CLI / Config YAML] --> D[Dataset Loader]
  D --> P[Preprocess + EDA]
  P --> T[Tokenizer Pipeline]
  T --> M[Model Loader]
  M --> TR[Trainer / Accelerate]
  TR --> E[Evaluation + Error Analysis]
  TR --> X[Export ONNX/TorchScript/Safetensors]
  E --> ML[MLflow Tracking]
  X --> I[Inference Engine]
  I --> API[FastAPI]
  I --> UI[Streamlit]
  I --> B[Benchmark Suite]
```

## Runtime Layers

1. Data Layer: load, validate, clean, dedupe, split, profile
2. Training Layer: full FT and PEFT workflows
3. Analytics Layer: metrics, explainability, error analysis, benchmarking
4. Serving Layer: API and UI with monitoring hooks
