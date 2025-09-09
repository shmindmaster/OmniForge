FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=1 \
  POETRY_VIRTUALENVS_CREATE=0

RUN apt-get update && apt-get install -y --no-install-recommends \
  libgl1 libglib2.0-0 && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY app /app/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8000"]
