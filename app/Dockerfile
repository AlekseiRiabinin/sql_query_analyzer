FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    python3.12-venv \
    libpq-dev \
    python3.12-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
