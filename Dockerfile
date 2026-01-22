FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y python3-venv gcc libffi-dev ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /venv && \
    /venv/bin/pip install --no-cache-dir --upgrade pip && \
    /venv/bin/pip install --no-cache-dir \
        langchain \
        langchain-openai \
        langchain-mongodb \
        langchain-text-splitters \
        pymongo \
        python-dotenv \
        certifi

COPY . .

ENTRYPOINT ["/venv/bin/python"]
