"""
Kronos inference service.

FastAPI wrapper around the OFFICIAL Kronos model.

Kronos inference runs in a worker thread so the FastAPI event loop
remains responsive while the model performs CPU/GPU inference.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model import Kronos, KronosTokenizer, KronosPredictor


TOKENIZER_ID = os.environ.get(
    "KRONOS_TOKENIZER",
    "NeoQuasar/Kronos-Tokenizer-base",
)

MODEL_ID = os.environ.get(
    "KRONOS_MODEL",
    "NeoQuasar/Kronos-small",
)

MAX_CONTEXT = int(
    os.environ.get("KRONOS_MAX_CONTEXT", "256")
)

DEVICE = os.environ.get("KRONOS_DEVICE")

API_TOKEN = os.environ.get("KRONOS_API_TOKEN")


app = FastAPI(
    title="Kronos Service",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


_state = {
    "status": "loading",
    "error": None,
    "predictor": None,
    "loaded_at": None,
}

_predict_lock = threading.Lock()


def _load() -> None:
    try:
        print("Loading Kronos tokenizer...", flush=True)

        tokenizer = KronosTokenizer.from_pretrained(
            TOKENIZER_ID
        )

        print("Loading Kronos model...", flush=True)

        model = Kronos.from_pretrained(
            MODEL_ID
        )

        kwargs = {
            "max_context": MAX_CONTEXT
        }

        if DEVICE:
            kwargs["device"] = DEVICE

        print(
            f"Creating KronosPredictor "
            f"(context={MAX_CONTEXT}, device={DEVICE or 'auto'})...",
            flush=True,
        )

        _state["predictor"] = KronosPredictor(
            model,
            tokenizer,
            **kwargs,
        )

        _state["loaded_at"] = time.time()
        _state["status"] = "ready"

        print("Kronos is ready.", flush=True)

    except Exception as exc:
        _state["status"] = "error"
        _state["error"] = f"{type(exc).__name__}: {exc}"

        print(
            f"Kronos loading error: {_state['error']}",
            flush=True,
        )


threading.Thread(
    target=_load,
    daemon=True,
).start()


class CandleIn(BaseModel):
    timestamp: int = Field(
        ...,
        description="epoch milliseconds (candle open time)",
    )
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: Optional[float] = None


class PredictRequest(BaseModel):
    candles: List[CandleIn]
    pred_len: int = 24
    interval_ms: int
    T: float = 1.0
    top_k: int = 0
    top_p: float = 0.9
    sample_count: int = 1


@app.get("/")
async def root():
    return {
        "service": "kronos",
        "status": _state["status"],
        "version": "2.0.0",
        "async_inference": True,
    }


@app.get("/health")
async def health():
    return {
        "status": _state["status"],
        "error": _state["error"],
        "model": MODEL_ID,
        "tokenizer": TOKENIZER_ID,
        "max_context": MAX_CONTEXT,
        "loaded_at": _state["loaded_at"],
        "async_inference": True,
    }


def _run_prediction(req: PredictRequest):
    """
    Executes the heavy Kronos inference in a worker thread.
    """

    rows = sorted(
        [c.model_dump() for c in req.candles],
        key=lambda r: r["timestamp"],
    )

    rows = rows[-MAX_CONTEXT:]

    df = pd.DataFrame(rows)

    df["amount"] = df.apply(
        lambda r:
            r["amount"]
            if r["amount"] is not None
            else r["volume"]
            * (
                r["open"]
                + r["high"]
                + r["low"]
                + r["close"]
            )
            / 4.0,
        axis=1,
    )

    x_timestamp = (
        pd.to_datetime(
            df["timestamp"],
            unit="ms",
            utc=True,
        )
        .dt.tz_localize(None)
    )

    x_df = df[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
        ]
    ].astype("float32")

    last_ts = int(
        df["timestamp"].iloc[-1]
    )

    future_ms = [
        last_ts + req.interval_ms * (i + 1)
        for i in range(req.pred_len)
    ]

    y_timestamp = pd.Series(
        pd.to_datetime(
            future_ms,
            unit="ms",
            utc=True,
        ).tz_localize(None)
    )

    started = time.time()

    with _predict_lock:

        predictor = _state["predictor"]

        if predictor is None:
            raise RuntimeError(
                "Kronos predictor is not loaded"
            )

        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=req.pred_len,
            T=req.T,
            top_k=req.top_k,
            top_p=req.top_p,
            sample_count=req.sample_count,
            verbose=False,
        )

    out = []

    for ts, row in zip(
        future_ms,
        pred_df.itertuples(index=False),
    ):
        out.append(
            {
                "timestamp": int(ts),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(row.volume),
            }
        )

    return {
        "predictions": out,
        "context_used": len(x_df),
        "model": MODEL_ID,
        "tokenizer": TOKENIZER_ID,
        "sample_count": req.sample_count,
        "inference_ms": int(
            (time.time() - started) * 1000
        ),
        "generated_at": int(
            time.time() * 1000
        ),
    }


@app.post("/predict")
async def predict(req: PredictRequest):

    if _state["status"] != "ready":
        raise HTTPException(
            status_code=503,
            detail=(
                f"Kronos not ready: "
                f"{_state['status']} "
                f"{_state['error'] or ''}"
            ),
        )

    if req.pred_len < 1 or req.pred_len > 240:
        raise HTTPException(
            status_code=400,
            detail="pred_len must be between 1 and 240",
        )

    if len(req.candles) < 32:
        raise HTTPException(
            status_code=400,
            detail="need at least 32 historical candles",
        )

    try:

        result = await asyncio.to_thread(
            _run_prediction,
            req,
        )

        return result

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Kronos prediction error: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc

