"""Serving — expose the model as a web API so other apps can use it.

Plain idea: wrap the `predict()` function in a tiny web server. Anyone can then
POST a car's details and get a price back over HTTP — no Python needed on their
side. Every prediction is also logged (monitoring.py) so we can watch quality.

Run locally:
    python -m car_pricing.serve            # http://localhost:8000
Production (inside the Docker image):
    gunicorn -w 2 -b 0.0.0.0:8000 car_pricing.serve:app

Endpoints:
    GET  /health   -> {"status": "healthy"}
    POST /predict  -> {"predicted_price_lakhs": ..., "price_band": ...}
"""

from __future__ import annotations

from flask import Flask, jsonify, request

from . import monitoring
from .predict import predict as _predict

app = Flask(__name__)


@app.get("/")
def index():
    return jsonify({
        "service": "car-pricing",
        "usage": {"method": "POST", "endpoint": "/predict",
                  "example": {"make": "MARUTI", "model": "SWIFT VXI",
                              "age": 5, "km_driven": 40000}},
    })


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


@app.post("/predict")
def do_predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON body, e.g. "
                        '{"make":"MARUTI","model":"SWIFT VXI"}'}), 400
    try:
        result = _predict(payload)
    except Exception as exc:                       # unknown make/model, bad input…
        return jsonify({"error": str(exc)}), 400
    monitoring.log_prediction(payload, result["predicted_price_lakhs"])
    return jsonify(result)


@app.errorhandler(405)
def method_not_allowed(_exc):
    return jsonify({"error": "Method not allowed. /predict expects POST."}), 405


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
