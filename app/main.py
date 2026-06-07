from fastapi import FastAPI, HTTPException
from datetime import datetime
from app.schemas import NetworkFlow, AnomalyResponse, StatsResponse
from app.ensemble import predict
from app.model import load_models

app = FastAPI(
    title="SIEM Anomaly Detection API",
    description="Real-time network traffic anomaly detection using Isolation Forest + Dense Autoencoder",
    version="1.0.0"
)

# In-memory stats
_stats = {
    "total_requests": 0,
    "total_anomalies": 0,
    "last_alert": None
}


@app.on_event("startup")
async def startup():
    """Load models saat server start."""
    load_models()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": ["isolation_forest", "dense_autoencoder"],
        "version": "1.0.0"
    }


@app.post("/ingest", response_model=AnomalyResponse)
def ingest(flow: NetworkFlow):
    """Terima network flow, return anomaly detection result."""
    global _stats

    try:
        raw = flow.model_dump(by_alias=True)
        result = predict(raw)

        _stats["total_requests"] += 1
        if result["is_anomaly"]:
            _stats["total_anomalies"] += 1
            _stats["last_alert"] = datetime.now().isoformat()

        return AnomalyResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
def stats():
    """Statistik deteksi real-time."""
    total = _stats["total_requests"]
    anomalies = _stats["total_anomalies"]

    return StatsResponse(
        total_requests=total,
        total_anomalies=anomalies,
        anomaly_rate=round(anomalies / total, 4) if total > 0 else 0.0,
        last_alert=_stats["last_alert"]
    )


@app.get("/")
def root():
    return {
        "service": "SIEM Anomaly Detection API",
        "docs": "/docs",
        "endpoints": ["/health", "/ingest", "/stats"]
    }