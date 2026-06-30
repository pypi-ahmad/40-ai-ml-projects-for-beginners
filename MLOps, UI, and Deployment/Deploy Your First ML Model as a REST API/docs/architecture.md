# API Architecture Diagrams

## 1. Request-Response Lifecycle

```mermaid
flowchart LR
    Client[Client Application] -->|HTTP Request| API[FastAPI Router]
    API --> Validation[Pydantic + Service Validation]
    Validation --> Predictor[Predictor Service]
    Predictor --> Model[(Joblib Model)]
    Model --> Predictor
    Predictor --> API
    API -->|JSON Response + Status + Request ID| Client
```

## 2. Inference + Explainability Path

```mermaid
flowchart TD
    A[/predict or /predict-batch/] --> B[Schema Validation]
    B --> C[Predictor.load() check]
    C --> D[Model.predict]
    D --> E[Response Schema]

    X[/explain/] --> Y[Schema Validation]
    Y --> Z[Predictor.explain_one]
    Z --> K[SHAP Explainer
Tree -> Linear -> Kernel fallback]
    K --> L[Feature Contributions JSON]
```

## 3. Middleware Stack (Outer -> Inner)

```mermaid
flowchart TD
    M1[SecurityHeadersMiddleware] --> M2[RequestIDMiddleware]
    M2 --> M3[RateLimitMiddleware]
    M3 --> M4[MetricsMiddleware]
    M4 --> M5[AccessLogMiddleware]
    M5 --> M6[APIKeyMiddleware]
    M6 --> M7[RequestSizeLimitMiddleware]
    M7 --> R[Route Handler]
```

## 4. Metrics and Observability

```mermaid
flowchart LR
    Req[Incoming Request] --> MM[MetricsMiddleware]
    MM --> DB[(SQLite api_metrics.db)]
    DB --> METRICS[/metrics endpoint/]
    METRICS --> DASH[Monitoring Consumer]
```
