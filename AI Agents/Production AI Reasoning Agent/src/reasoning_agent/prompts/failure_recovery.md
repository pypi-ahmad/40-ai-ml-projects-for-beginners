Failure recovery policy:
- Never fabricate successful tool output.
- Retry only for transient errors (timeouts, parsing noise, network instability).
- On deterministic validation/security errors, stop retries and produce safe fallback.
