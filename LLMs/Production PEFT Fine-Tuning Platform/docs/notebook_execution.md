# Notebook Execution

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/peft_zero_to_hero.ipynb --output peft_zero_to_hero.executed.ipynb
```

If GPU unavailable, notebook still runs in CPU/mock path for smoke demonstration.
