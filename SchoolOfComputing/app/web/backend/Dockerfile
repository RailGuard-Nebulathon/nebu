FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY web/backend ./web/backend
RUN pip install --no-cache-dir -e ".[api]"

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "railguard_api.main:app", "--app-dir", "web/backend", "--host", "0.0.0.0", "--port", "8000"]
