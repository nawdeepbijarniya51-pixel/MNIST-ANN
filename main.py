from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = FastAPI()

# Allow your HTML frontend to call this API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your model once at startup
model = tf.keras.models.load_model("mnist_ann.keras")

@app.get("/")
def root():
    return {"status": "MNIST API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("L")  

    # Resize to 28x28 (what MNIST expects)
    img = img.resize((28, 28))

    # Normalize to 0-1 range, flatten to (1, 784)
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, -1)   # ANN expects flat input

    # Get predictions (softmax probabilities)
    probs = model.predict(img_array)[0].tolist()
    predicted_digit = int(np.argmax(probs))

    return {
        "predicted_digit": predicted_digit,
        "probabilities": probs
    }