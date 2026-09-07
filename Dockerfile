FROM python:3.10-slim

WORKDIR /srv

# Instala PyTorch somente para CPU.
# O PyPI continua disponível para as demais dependências.
RUN pip install --no-cache-dir torch \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000
ENV KRONOS_MODEL=NeoQuasar/Kronos-small
ENV KRONOS_TOKENIZER=NeoQuasar/Kronos-Tokenizer-base
ENV KRONOS_MAX_CONTEXT=256

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]