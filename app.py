from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from typing import Optional
import os
import sys

app = FastAPI(
    title="Car Price Prediction API",
    description="API для оценки стоимости автомобиля по его характеристикам",
    version="1.0.0"
)

try:
    model = joblib.load('models/best_model.joblib')
    preprocessor = joblib.load('models/preprocessor.joblib')
except Exception as e:
    model = None
    preprocessor = None


class CarFeatures(BaseModel):
    name: str
    year: int
    mileage: float
    engine_volume: float
    horse_power: float
    city: str


class PredictionResponse(BaseModel):
    predicted_price: int
    confidence: str
    model_info: dict

@app.get("/")
async def root():
    return {
        "message": "Car Price Prediction API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
        }
    }

@app.get("/health")
async def health_check():
    return {
        "health": "healthy" if model and preprocessor else "unhealthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }

if __name__ == "__main__":
    import uvicorn
    print("move to: http://localhost:8000/")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
