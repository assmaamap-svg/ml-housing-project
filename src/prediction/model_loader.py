import joblib
from src.prediction.config import MODEL_PATH


def load_model():
    """Charge le modèle depuis le chemin défini dans config."""
    return joblib.load(MODEL_PATH)