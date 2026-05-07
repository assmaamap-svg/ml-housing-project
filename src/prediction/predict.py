import pandas as pd
from src.prediction.model_loader import load_model

model = load_model()


def predict(data: dict) -> float:
    """Prend un dict de features et retourne une prédiction."""
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return float(prediction)
