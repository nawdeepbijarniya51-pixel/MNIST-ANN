from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from PIL import ImageOps

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # allows ALL origins including GitHub Pages
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("mnist_cnn.keras")

@app.get("/")
def root():
    return {"status": "MNIST API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    # 1. Open and convert to grayscale
    img = Image.open(io.BytesIO(contents)).convert("L")
    
    # 2. Thresholding: Make it strictly binary (0 or 255)
    # This helps remove "gray" noise and shadows
    threshold = 127
    img = img.point(lambda p: 255 if p > threshold else 0)
    
    # 3. Invert colors: If your background is white and digit is black, 
    # MNIST requires white digits on black background.
    # Note: If your image already has black background, comment this out.

    img = ImageOps.invert(img)
    
    # 4. Resize and convert to array
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    probs = model.predict(img_array)[0].tolist()
    predicted_digit = int(np.argmax(probs))
    
    return {
        "predicted_digit": predicted_digit,
        "probabilities": probs
    }
