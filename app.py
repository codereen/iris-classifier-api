"""FastAPI service for classifying Iris flowers."""

from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field


MODEL_PATH = Path(__file__).resolve().parent / "models" / "iris_classifier.joblib"
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
target_names = bundle["target_names"]

PositiveMeasurement = Annotated[float, Field(gt=0, le=10)]


class FlowerMeasurements(BaseModel):
    """Measurements are in centimeters and must be positive, plausible values."""

    sepal_length: PositiveMeasurement = Field(description="Sepal length in cm")
    sepal_width: PositiveMeasurement = Field(description="Sepal width in cm")
    petal_length: PositiveMeasurement = Field(description="Petal length in cm")
    petal_width: PositiveMeasurement = Field(description="Petal width in cm")


class Prediction(BaseModel):
    species: str
    class_id: int
    confidence: float


app = FastAPI(
    title="Iris Classifier API",
    description="Predict an Iris flower species from four measurements in centimeters.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(flower: FlowerMeasurements) -> Prediction:
    values = np.array(
        [[
            flower.sepal_length,
            flower.sepal_width,
            flower.petal_length,
            flower.petal_width,
        ]]
    )
    probabilities = model.predict_proba(values)[0]
    class_id = int(np.argmax(probabilities))
    return Prediction(
        species=target_names[class_id],
        class_id=class_id,
        confidence=float(probabilities[class_id]),
    )
