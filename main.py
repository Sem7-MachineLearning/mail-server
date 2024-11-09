from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import matplotlib.pyplot as plt
import string
import random
from fastapi.staticfiles import StaticFiles

vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl') 

class InputData(BaseModel):
    subject: str
    body: str

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'^subject:\s*', '', text)
    text = re.sub(r'\W', ' ', text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

app.mount("/result", StaticFiles(directory="result"), name="result")

@app.get("/status")
def status():
    return {"status": "ok"}

@app.post("/predict")
def predict(input_data: InputData):
    email = input_data.subject + " " + input_data.body
    email = preprocess_text(email)
    features = vectorizer.transform([email])
    prediction_proba = model.predict_proba(features)[0]
    prediction = model.predict(features)

    image_name = ''.join(random.choices(string.ascii_letters, k=7))

    labels = ['Not Spam', 'Spam']
    colors = ['lightblue', 'salmon']
    plt.figure(figsize=(6, 6))
    plt.pie(prediction_proba, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title("Probability Distribution for Email Classification")
    plt.savefig("result/" + image_name + ".png")

    result = bool(prediction[0])

    return {
        "prediction": result,
        "image_name": image_name,
    }
