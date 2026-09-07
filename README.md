# Kronos Service (external Python service)

The Kronos model is PyTorch-based and **cannot run inside this web app's runtime**
(the app backend runs on an edge/serverless runtime with no Python). This folder
contains the service you must host separately. The web app talks to it over HTTP.

```
Binance -> app backend (candles) -> THIS SERVICE (official Kronos) -> forecast -> dashboard
```

## 1. Get the official Kronos code

```bash
cd kronos-service
git clone https://github.com/shiyu-coder/Kronos.git .kronos-src
cp -r .kronos-src/model ./model      # model/__init__.py, model/kronos.py, model/module.py
```

Final layout:

```
kronos-service/
  app.py            <- this wrapper (do not modify the Kronos code)
  requirements.txt
  model/            <- OFFICIAL Kronos package goes HERE
    __init__.py
    kronos.py
    module.py
```

## 2. Weights / checkpoints

Weights are pulled automatically from HuggingFace on first start
(`from_pretrained`). Official checkpoints:

| Model          | Tokenizer                        | Context |
| -------------- | -------------------------------- | ------- |
| Kronos-mini    | NeoQuasar/Kronos-Tokenizer-2k    | 2048    |
| Kronos-small   | NeoQuasar/Kronos-Tokenizer-base  | 512     |
| Kronos-base    | NeoQuasar/Kronos-Tokenizer-base  | 512     |

If you prefer local weights, place them anywhere and point
`KRONOS_MODEL` / `KRONOS_TOKENIZER` at the local directory paths.

## 3. Install and run

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Environment variables:

| Var                 | Default                          |
| ------------------- | -------------------------------- |
| KRONOS_MODEL        | NeoQuasar/Kronos-small           |
| KRONOS_TOKENIZER    | NeoQuasar/Kronos-Tokenizer-base  |
| KRONOS_MAX_CONTEXT  | 512                              |
| KRONOS_DEVICE       | auto (cuda / mps / cpu)          |

## 3.1 Render grátis (512 MB de RAM) — versão enxuta

O PyTorch comum não cabe no plano grátis do Render. O `Dockerfile` desta pasta
já está adaptado:

- **PyTorch somente CPU** (sem CUDA) — reduz muito o tamanho e a RAM.
- **Kronos-mini** (`NeoQuasar/Kronos-mini` + `NeoQuasar/Kronos-Tokenizer-2k`),
  o menor modelo oficial, feito para CPU.
- 1 único worker e 1 thread (`OMP_NUM_THREADS=1`).

No Render: crie um **Web Service** a partir do repositório, aponte o
*Root Directory* para `kronos-service`, runtime **Docker**, e pronto — os pesos
são baixados do HuggingFace no primeiro start (leva alguns minutos).

Se ainda assim faltar memória, as alternativas são Fly.io, Railway ou um VPS
pequeno com 1–2 GB de RAM.

## 4. Connect it to the dashboard

Expose the service on a public HTTPS URL (Render, Railway, Fly.io, RunPod, your
own VPS behind a reverse proxy). Then in the dashboard open **Kronos service**
in the side panel and paste the base URL (e.g. `https://kronos.example.com`).

The app calls:

- `GET  /health`  -> `{ status: "loading" | "ready" | "error", model, max_context }`
- `POST /predict` -> `{ candles[], pred_len, interval_ms, T, top_k, top_p, sample_count }`

`/predict` returns the raw denormalized Kronos output candles
(`open, high, low, close, volume`) with future timestamps.

No trading endpoints exist and no Binance API key is ever used.
