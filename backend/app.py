from fastapi import FastAPI
from src.prediction.predict import predict
from src.prediction.schemas import HousingFeatures

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_price(data: HousingFeatures):
    result = predict(data.model_dump())
    return {"prediction": result}