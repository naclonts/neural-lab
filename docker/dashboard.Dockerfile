FROM python:3.11-slim-bookworm
COPY neural_lab/dashboard.py /app/neural_lab/dashboard.py
COPY neural_lab/web /app/neural_lab/web
ENV PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
ENTRYPOINT ["python3", "-m", "neural_lab.dashboard"]
