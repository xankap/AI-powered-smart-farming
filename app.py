from fastapi import FastAPI, File, UploadFile
import numpy as np
import onnxruntime as ort
from PIL import Image
import io

app = FastAPI()

# Load ONNX model
session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((640, 640))  # match training size

    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))  # CHW
    img_array = np.expand_dims(img_array, axis=0)

    outputs = session.run(None, {input_name: img_array})

    return {"predictions": outputs}
