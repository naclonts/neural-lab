FROM python:3.11-slim-bookworm
COPY --from=docker:29-cli /usr/local/bin/docker /usr/local/bin/docker
WORKDIR /repo
COPY neural_lab /opt/neural_lab
ENV PYTHONPATH=/opt
ENTRYPOINT ["python3", "-m", "neural_lab.queue"]
