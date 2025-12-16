from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image
import io, uuid, threading

app = Flask(__name__)

# Load model once
session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# Store results
jobs = {}

# -------------------------------
# BACKGROUND INFERENCE FUNCTION
# -------------------------------
def run_inference(job_id, img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((640, 640))

        img_array = np.array(img).astype(np.float32)
        img_array = img_array[:, :, ::-1] / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)

        outputs = session.run(None, {input_name: img_array})

        # Example: simple ripe logic
        confidence = float(np.max(outputs[0]))
        ripe = confidence > 0.6

        jobs[job_id] = {
            "status": "done",
            "ripe": ripe,
            "confidence": confidence
        }

    except Exception as e:
        jobs[job_id] = {
            "status": "error",
            "error": str(e)
        }

# -------------------------------
# IMAGE UPLOAD ENDPOINT
# -------------------------------
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]
    img_bytes = file.read()

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing"}

    # Run inference in background
    threading.Thread(
        target=run_inference,
        args=(job_id, img_bytes)
    ).start()

    # INSTANT ACK to ESP32
    return jsonify({
        "status": "received",
        "job_id": job_id
    })

# -------------------------------
# RESULT ENDPOINT
# -------------------------------
@app.route("/result/<job_id>", methods=["GET"])
def result(job_id):
    return jsonify(jobs.get(job_id, {"status": "not_found"}))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
