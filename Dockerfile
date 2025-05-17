# Етап збірки
FROM python:slim as builder

# Встановлюємо системні залежності для збірки
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app
COPY requirements.lock ./
RUN uv pip install --no-cache --system -r requirements.lock

# Фінальний етап
FROM python:slim

# Копіюємо лише необхідні бібліотеки з етапу збірки
COPY --from=builder /usr/local/lib/python3.*/site-packages /usr/local/lib/python3.*/site-packages

WORKDIR /app
COPY src/coursework_operations .
CMD ["python", "main.py"]