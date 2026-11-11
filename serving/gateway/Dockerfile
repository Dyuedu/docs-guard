FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"
COPY serving/gateway ./serving/gateway
RUN touch serving/__init__.py

EXPOSE 8080
CMD ["uvicorn", "serving.gateway.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
