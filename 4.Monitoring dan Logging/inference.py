import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import start_http_server, Summary, Counter, Gauge
import time
import uvicorn

# === Metrik Prometheus ===
REQUEST_TIME = Summary('inference_request_duration_seconds', 'Time for a prediction request')
TOTAL_REQUESTS = Counter('inference_total_requests', 'Total prediction requests')
FAILED_REQUESTS = Counter('inference_failed_requests', 'Total failed prediction requests')
CURRENT_LATENCY = Gauge('inference_current_latency_ms', 'Latest latency in milliseconds')
MODEL_OUTPUT_CLASS = Gauge('inference_output_class', 'Predicted output class (numeric)')

# === Load model ===
model = joblib.load("model.pkl")
print("✅ Model loaded.")

# === FastAPI App ===
app = FastAPI()

class InferenceRequest(BaseModel):
    dataframe_split: dict

@app.get("/")
def root():
    return {"message": "Model is ready for inference."}

@app.post("/predict")
@REQUEST_TIME.time()
def predict(request: InferenceRequest):
    TOTAL_REQUESTS.inc()
    start = time.time()

    try:
        df = pd.DataFrame(**request.dataframe_split)
        pred = model.predict(df)
        latency = (time.time() - start) * 1000  # ms
        CURRENT_LATENCY.set(latency)
        MODEL_OUTPUT_CLASS.set(pred[0])

        return {"prediction": int(pred[0]), "latency_ms": latency}

    except Exception as e:
        FAILED_REQUESTS.inc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Start Prometheus Exporter
    start_http_server(8001)  # <-- port Prometheus
    # Start FastAPI inference server
    uvicorn.run(app, host="0.0.0.0", port=8000)