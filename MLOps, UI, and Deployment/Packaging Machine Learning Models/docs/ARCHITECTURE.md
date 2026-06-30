# Packaging Architecture

## Runtime Layers

1. Client layer
2. FastAPI service layer (`api/main.py`)
3. Wrapper layer (`PredictionEngine`, `ModelLoader`, validators)
4. Serialized artifact layer (`models/*.pkl|joblib|onnx|pt` + manifests)
5. Registry/version layer (`models/registry.json`)

## Core Contracts

- Input contract: strict Pydantic schemas (`api/schemas.py`)
- Prediction contract: structured payload from `PredictionEngine`
- Security contract: manifest + trusted digest checks before unsafe deserialization
- Version contract: active version + lineage metadata in registry

## Main Flows

### Training and packaging flow

`train_model.py -> benchmark -> serialize -> write manifests -> register v1/v2 -> activate v2`

### Serving flow

`request -> schema validation -> prediction engine -> metrics/logs -> response`

### Explainability flow

`/explain -> ModelExplainer -> SHAP local/global output`
