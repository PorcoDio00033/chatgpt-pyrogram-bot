# build phase
FROM python:3.13-slim as builder

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    PIP_DEFAULT_TIMEOUT=100

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# should be fine with pip in builder, actually i'm not too sure lol
RUN pip install "poetry==2.2.1"

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-root --only main


# run phase
FROM python:3.13-slim as runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# ffmpeg: required for audio processing
# cairo: required for converting tsg (animated telegram stickers) to mp4
RUN apt-get update && apt-get install -y \
    ffmpeg=7:7.1.3-0+deb13u1 \
    curl \
    gnupg \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 -U appuser

COPY --from=builder /app/.venv ./.venv

COPY . .

RUN mkdir -p /app/data && chown -R appuser:appuser /app/data

USER appuser

CMD ["python", "bot/main.py"]