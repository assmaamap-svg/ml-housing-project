from fastapi import FastAPI
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI()


def get_latest_model():
    """Trouve et charge le modèle avec le numéro de version le plus élevé."""
    artifacts_path = Path("artifacts")
    models = list(artifacts_path.glob("model_v*.joblib"))
    if not models:
        default_model = artifacts_path / "model.joblib"
        if default_model.exists():
            return joblib.load(default_model)
        raise FileNotFoundError("Aucun modèle trouvé dans le dossier artifacts.")
    latest_model_path = sorted(models, key=lambda x: int(x.stem.split("_v")[-1]))[-1]
    return joblib.load(latest_model_path)


model = get_latest_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return {"prediction": float(prediction)}
