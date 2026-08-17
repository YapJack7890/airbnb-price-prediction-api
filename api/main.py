from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import json
import numpy as np
import joblib



app = FastAPI(
    title="Airbnb Price Prediction API",
    version="1.0.0"
)

# Project root: /project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Model: /project/models/stacking_regressor.pkl
MODEL_PATH = PROJECT_ROOT / "models" / "stacking_regressor.pkl"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

# ============================================================
# Load model
# ============================================================
model = joblib.load(MODEL_PATH)

# ============================================================
# Load neighbourhood mapping
# ============================================================
MODELS_DIR = PROJECT_ROOT / "models"

with open(MODELS_DIR / "neighbourhood_mapping.json", "r", encoding="utf-8") as f:
    neighbourhood_mapping = json.load(f)

with open(MODELS_DIR / "encoder_maps.json", "r", encoding="utf-8") as f:
    encoder_maps = json.load(f)

# ============================================================
# Request schema
# ============================================================

class PredictionRequest(BaseModel):
    host_is_superhost: int
    neighbourhood: str
    latitude: float
    longitude: float
    property_type: str
    room_type: str
    accommodates: int
    bathrooms: float
    bedrooms: float
    beds: float
    minimum_nights: int
    availability_365: int
    instant_bookable: int


# ============================================================
# Helper: convert neighbourhood -> neighbourhood group
# ============================================================

def get_neighbourhood_group(neighbourhood: str) -> str:
    for group, neighbourhoods in neighbourhood_mapping.items():
        if neighbourhood in neighbourhoods:
            return group

    raise ValueError(
        f"Unknown neighbourhood: {neighbourhood}"
    )


def get_encoded_value(value: str, key: str) -> int:
    mapping = encoder_maps.get(key)
    for map_key, encoded_value in mapping.items():
        if value == map_key:
            return encoded_value

    raise ValueError(
        f"Unknown value: {value}"
    )


# ============================================================
# Feature preparation
# ============================================================

def prepare_features(request: PredictionRequest):

    neighbourhood_group = get_neighbourhood_group(
        request.neighbourhood
    )

    neighbourhood = get_encoded_value(
        request.neighbourhood, "neighbourhood_cleansed"
    )

    property_type = get_encoded_value(
        request.property_type, "property_type"
    )

    # Start with raw features
    data = {
        "host_is_superhost": request.host_is_superhost,
        "neighbourhood": neighbourhood,
        "latitude": request.latitude,
        "longitude": request.longitude,
        "property_type": property_type,
        "room_type": request.room_type,
        "accommodates": request.accommodates,
        "bathrooms": request.bathrooms,
        "bedrooms": request.bedrooms,
        "beds": request.beds,
        "minimum_nights": request.minimum_nights,
        "availability_365": request.availability_365,
        "instant_bookable": request.instant_bookable,

        # Derived feature
        "neighbourhood_group_cleansed": neighbourhood_group,
    }

    df = pd.DataFrame([data])

    # --------------------------------------------------------
    # One-hot encode room_type
    # --------------------------------------------------------

    room_types = [
        "Entire home/apt",
        "Hotel room",
        "Private room",
        "Shared room"
    ]

    for room_type in room_types:
        df[f"room_type_{room_type}"] = (
            df["room_type"] == room_type
        ).astype(int)

    # --------------------------------------------------------
    # One-hot encode neighbourhood_group_cleansed
    # --------------------------------------------------------

    neighbourhood_groups = [
        "City of Los Angeles",
        "Other Cities",
        "Unincorporated Areas"
    ]

    for group in neighbourhood_groups:
        df[
            f"neighbourhood_group_cleansed_{group}"
        ] = (
            df["neighbourhood_group_cleansed"] == group
        ).astype(int)

    # Remove original room_type because model doesn't expect it
    df.drop(columns=["room_type"], inplace=True)

    # Remove original neighbourhood group because model
    # expects the one-hot columns
    df.drop(
        columns=["neighbourhood_group_cleansed"],
        inplace=True
    )

    df = df.rename(columns={
        "neighbourhood": "neighbourhood_cleansed"
    })

    return df


# ============================================================
# Prediction
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        X = prepare_features(request)

        prediction = model.predict(X)

        predicted_value = float(prediction[0])

        # If model predicts log(price)
        predicted_price = np.expm1(predicted_value)

        return {
            "predicted_price": round(predicted_price, 2)
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# ============================================================
# Health check
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }