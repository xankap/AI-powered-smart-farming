

from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import os
# Initialize Flask app
app = Flask(__name__)
# Load YOLO model (replace with your model path)
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

print(f"✅ Ultralytics model loaded from {MODEL_PATH}")
# Endpoint to test API status
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Tomato Ripeness Detection API is running 🚀",
        "usage": "POST an image to /predict"
    })
# Endpoint for image prediction
@app.route("/predict", methods=["POST"])
def predict():
    # Check if an image was sent
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    # Run inference with YOLO
    results = model.predict(image, imgsz=640, conf=0.25)
    # Parse results
    predictions = []
    for box in results[0].boxes:
        cls_id = int(box.cls)
        label = results[0].names[cls_id]
        confidence = float(box.conf)
        predictions.append({
            "label": label,
            "confidence": round(confidence, 3)
        })
    # Return response
    return jsonify({
        "detections": predictions,
        "num_detections": len(predictions)
    })
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Allow cloud platforms to assign a port
    app.run(host="0.0.0.0", port=port)