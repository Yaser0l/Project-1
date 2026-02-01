import sys
import re
import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from nltk.stem import WordNetLemmatizer

# 1. Standard Definitions
lemmatizer = WordNetLemmatizer()
stop_words = set(['the', 'is', 'at', 'which', 'on']) 

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r"\d+", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def pipeline_preprocess(text_series):
    if isinstance(text_series, str):
        text_series = [text_series]
    return [preprocess_text(text) for text in text_series]

# --- THE CRITICAL FIX ---
# We manually inject the function into the modules where joblib might look
import __main__
__main__.pipeline_preprocess = pipeline_preprocess
sys.modules['pipeline_preprocess'] = pipeline_preprocess 
# ------------------------

app = FastAPI(title="Genre Classifier API")

# Path setup
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "notebooks", "SVC_pipeline.joblib")

# Load model
model = joblib.load(model_path)

class BookInput(BaseModel):
    text: str

@app.get("/")
def health_check():
    return {"status": "Online", "model": "SVC_pipeline"}

@app.post("/predict")
def predict_genre(data: BookInput):
    # Pipeline handles [preprocess -> tfidf -> svc] automatically
    prediction = model.predict([data.text])[0]
    genre = "Fiction" if prediction == 1 else "Nonfiction"
    
    return {
        "input_text": data.text,
        "prediction": genre,
        "label_id": int(prediction)
    }