# Production image for the serving API (car_pricing.serve:app via gunicorn).
# Build:  docker build -t car-pricing-api .
# Run:    docker run -p 8000:8000 car-pricing-api
# Test:   curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
#              -d '{"make":"MARUTI","model":"SWIFT VXI","age":5,"km_driven":40000}'
FROM python:3.11-slim

WORKDIR /app

# Install deps first so this layer is cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Install the package, then copy the model artifacts it serves.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .
COPY models/ ./models/

EXPOSE 8000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "car_pricing.serve:app"]
