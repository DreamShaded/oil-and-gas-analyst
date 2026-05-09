# syntax=docker/dockerfile:1.6

FROM python:3.12-slim AS deps

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev


FROM python:3.12-slim AS runtime

RUN useradd --create-home --uid 1000 app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright

WORKDIR /app

COPY --from=deps /opt/venv /opt/venv

# Системные зависимости Chromium + один раз ставим headless-shell для Playwright
RUN apt-get update && apt-get install -y --no-install-recommends \
        libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
        libdbus-1-3 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 \
        libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2 libatspi2.0-0 \
        libx11-xcb1 fonts-liberation \
    && rm -rf /var/lib/apt/lists/* \
    && /opt/venv/bin/python -m playwright install chromium \
    && chown -R app:app /opt/ms-playwright

COPY --chown=app:app . .

USER app

EXPOSE 7860

CMD ["python", "main.py", "serve"]
