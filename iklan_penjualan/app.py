# Import library FastAPI
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# Inisialisasi FastAPI
app = FastAPI()

# Load model dari file
with open("models/linear_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Buat schema input dari user (request body)
class InputData(BaseModel):
    tv: float
    radio: float
    newspaper: float

# Endpoint untuk prediksi
@app.post("/predict")
def predict(data: InputData):
    # Ambil data input dan ubah ke array 2 dimensi
    x_input = np.array([[data.tv, data.radio, data.newspaper]])

    # Prediksi menggunakan model yang sudah dilatih
    prediction = loaded_model.predict(x_input)

    return {
        "TV": data.tv,
        "Radio": data.radio,
        "Newspaper": data.newspaper,
        "Predicted Sales": prediction[0][0]
    }
