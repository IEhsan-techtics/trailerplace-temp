# The chatbot API. One image, one process: uvicorn serving main:app.
#
# Built for Azure Container Apps, where the container is stopped by SIGTERM and may be
# scaled to zero between customers. Two consequences shape this file: the signal has to
# reach uvicorn (see CMD), and a cold start has to be survivable, which is why /health
# warms the caches rather than merely answering.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# curl and ca-certificates only: psycopg[binary] and the openai SDK ship their own wheels,
# so there is no compiler here and nothing to build.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Its own layer, ahead of the source: the dependency install is the slow half of a build
# and it should only rerun when the dependencies actually change.
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt

COPY . .

# Not root. LOG_DIR defaults to ./logs and the app writes a file per day, so that directory
# has to exist and be owned by the user that will write it - created here rather than left
# to the first request, which would fail on a read-only root filesystem.
RUN useradd --create-home --uid 10001 trailerplace \
    && mkdir -p /app/logs \
    && chown -R trailerplace:trailerplace /app
USER trailerplace

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=8)"

# Exec form, so uvicorn is PID 1 and receives SIGTERM directly. It matters here: the
# lifespan shutdown calls turn_saver.drain(), and a turn that was answered but whose commit
# is still queued is lost if the process is killed instead of asked to stop.
#
# The graceful window is 30s against a 25s drain timeout, so the drain gets to finish - or
# to log the turns it could not save, which is the whole point of having it.
CMD ["python", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-graceful-shutdown", "30"]
