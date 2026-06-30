# Prompt Flow

```mermaid
flowchart TD
    Q[User Question] --> G[Glossary Matcher]
    G --> S[Schema Context Builder]
    S --> P[Persona Template]
    P --> M[Memory Context]
    M --> Prompt[Final Prompt]
    Prompt --> LLM[Ollama Model]
    LLM --> SQL[Raw SQL Candidate]
    SQL --> Cleaner[SQL Cleaner + Formatter]
```
