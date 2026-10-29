FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends docker.io \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /repo
COPY neural_lab /opt/neural_lab
ENV PYTHONPATH=/opt
ENTRYPOINT ["python3", "-m", "neural_lab.queue"]
