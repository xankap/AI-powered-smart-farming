from fastapi import FastAPI, UploadFile, File
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = FastAPI()

# -------------------------
# LOAD ONNX MODEL
# -------------------------
MODEL_PATH = "best.onnx"

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name


# -------------------------
# PREDICT ENDPOINT
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        img_bytes = await file.read()

        # Load image
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((640, 640))

        # Convert to array
        img_array = np.array(img).astype(np.float32)

        # Convert RGB → BGR (YOLO ONNX expects BGR)
        img_array = img_array[:, :, ::-1]

        # Normalize
        img_array /= 255.0

        # CHW format
        img_array = np.transpose(img_array, (2, 0, 1))

        # Add batch dim
        img_array = np.expand_dims(img_array, axis=0)

        # Run inference
        outputs = session.run(None, {input_name: img_array})

        # Convert outputs to python lists
        predictions = [out.tolist() for out in outputs]

        return {"success": True, "predictions": predictions}

    except Exception as e:
        return {"success": False, "error": str(e)}
