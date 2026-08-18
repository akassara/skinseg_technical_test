FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    nnUNet_raw=/workspace/nnunet_data/nnunet_raw \
    nnUNet_preprocessed=/workspace/nnunet_data/nnunet_preprocessed \
    nnUNet_results=/workspace/nnunet_data/nnunet_results

WORKDIR /workspace/skinseg

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY nnUNet/ ./nnUNet/
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/

RUN python -m pip install --upgrade pip \
    && python -m pip install ./nnUNet . \
    && chmod +x ./scripts/launch.sh

VOLUME ["/workspace/data", "/workspace/nnunet_data"]

ENTRYPOINT ["./scripts/launch.sh"]
