FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

# Install project dependencies
RUN pip install --no-cache-dir \
  --retries 10 \
  --timeout 120 \
  .

COPY app ./app

COPY certs ./certs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]