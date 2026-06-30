# Visualization Pipeline

```mermaid
flowchart LR
    ResultRows[Execution Rows] --> Analyzer[Type + Cardinality Analyzer]
    Analyzer --> Recommend[Chart Recommender]
    Recommend --> Options[table/bar/line/pie/scatter/hist/heatmap/time-series]
    Options --> Renderer[Plotly Renderer]
    Renderer --> UI[Streamlit UI]
```
