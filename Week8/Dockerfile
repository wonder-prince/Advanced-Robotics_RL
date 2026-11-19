FROM python:3.10.12-slim-bookworm
RUN pip install --no-cache-dir uv==0.8.16
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libasound2 libgl1 libglib2.0-0 libgtk-3-0 libnss3 libx11-6 libxcursor1 libxi6 libxrandr2 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /lab
COPY .python-version pyproject.toml uv.lock ./
RUN uv sync --locked
COPY train.py ./
COPY config/ config/
ENTRYPOINT ["/lab/.venv/bin/python", "/lab/train.py"]
