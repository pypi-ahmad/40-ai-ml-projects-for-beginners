FROM python:3.12.10-slim

WORKDIR /app

# Naive teaching image: installs full training + notebook + serving dependencies.
RUN python -m pip install --no-cache-dir uv
COPY requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

COPY app/ app/
COPY models/ models/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
