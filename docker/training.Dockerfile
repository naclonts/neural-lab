FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
# Data-only Hub loading; datasets >=4 does not execute repository loading scripts.
RUN python -m pip install --no-cache-dir --timeout=120 --retries=5 --resume-retries=5 \
    datasets==4.8.5 huggingface_hub==1.3.7 pyarrow==21.0.0 pandas==2.3.3
ENV HF_HOME=/tmp/huggingface
