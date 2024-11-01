from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl') 

class InputData(BaseModel):
    subject: str
    body: str

app = FastAPI()

@app.get("/status")
def status():
    return {"status": "ok"}

@app.post("/predict")
def predict(input_data: InputData):
    features = vectorizer.transform([input_data.subject + " " + input_data.body])
    prediction = model.predict(features)
    result = bool(prediction[0])
    return {"prediction": result}
