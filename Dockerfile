FROM python:3.10-slim

WORKDIR /srv

# PyTorch SOMENTE CPU (muito menor que a versão com CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# A pasta oficial `model/` do Kronos deve estar junto do app.py.
COPY . .

# Kronos-mini: modelo menor, feito para rodar em CPU com pouca RAM.
# Limita threads para não estourar memória no plano grátis (512 MB).
ENV KRONOS_MODEL=NeoQuasar/Kronos-mini \
    KRONOS_TOKENIZER=NeoQuasar/Kronos-Tokenizer-2k \
    KRONOS_MAX_CONTEXT=1024 \
    KRONOS_DEVICE=cpu \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1

EXPOSE 8000
# 1 worker apenas — cada worker extra duplicaria o modelo em memória.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
