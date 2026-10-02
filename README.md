# ML Backend Service

Data ingestion, inference, and evaluation pipeline for ML workloads.

## Structure

- **service/inference.py**: Model serving endpoint
- **service/data_ingestion.py**: Data preparation and normalization
- **service/evaluation.py**: Model evaluation and scoring
- **service/app.py**: FastAPI application

## Setup

```bash
pip install -r requirements.txt
uvicorn service.app:app --reload
```

## Endpoints

- `POST /predict`: Get model predictions for input features
- `POST /evaluate`: Score predictions against ground truth
- `GET /health`: Health check
