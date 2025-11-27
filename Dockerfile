FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/data/hf_cache \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    POETRY_VERSION=2.0.0 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 

ENV PATH="$POETRY_HOME/bin:$PATH"

RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    curl \
    build-essential \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /workspace

COPY pyproject.toml poetry.lock* ./

RUN poetry install --no-root --only main

COPY . .

RUN mkdir -p /workspace/data

ENV PYTHONPATH=.

ENTRYPOINT ["poetry", "run", "python", "projected_token/cli.py"]

CMD ["--help"]
