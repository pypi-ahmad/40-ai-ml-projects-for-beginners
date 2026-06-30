FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY configs ./configs
COPY src ./src
COPY streamlit_app ./streamlit_app
COPY app.py main.py ./

RUN pip install uv && uv sync --frozen || uv sync

EXPOSE 8000 8501
CMD ["uv", "run", "python", "main.py"]
