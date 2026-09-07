FROM python:3.10-slim

WORKDIR /srv
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The official Kronos `model/` package must be present next to app.py.
COPY . .

ENV KRONOS_MODEL=NeoQuasar/Kronos-small \
    KRONOS_TOKENIZER=NeoQuasar/Kronos-Tokenizer-base \
    KRONOS_MAX_CONTEXT=512

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
