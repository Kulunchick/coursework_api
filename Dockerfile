# Етап збірки
FROM python:slim AS builder

# Встановлюємо системні залежності для збірки
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN uv venv .venv
COPY requirements.lock ./
RUN uv pip install --no-cache -r requirements.lock

# Фінальний етап
FROM python:slim

# Копіюємо лише необхідні бібліотеки з етапу збірки
COPY --from=builder /app/.venv /app/.venv

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"
COPY src src
CMD ["python", "-m", "src.coursework_operations.main"]