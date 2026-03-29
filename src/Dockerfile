FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Requirements are copied from project root (build context = project root)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and config
COPY src/ /app/src/
COPY config/ /app/config/

RUN mkdir -p /app/data/models

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
