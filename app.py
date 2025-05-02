import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app, supports_credentials=True)
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def process_image(image_file):
    """
    Convert the image to a format suitable for sending to the Gemini API.
    """
    image = Image.open(image_file)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return base64_image


@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    base64_image = process_image(image_file)

    payload = {
        "contents": [{
            "parts": [
                {
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": base64_image
                    }
                },
                {
                    "text": (
                        "You are an expert grievance classification system. Given an input image showing an issue, you must respond with three things:\n\n"
                        "1. Title: A short, 3-6 word heading describing the grievance.\n"
                        "2. Description: A short, 2-3 sentence description about the issue shown.\n"
                        "3. Category: Choose the most appropriate department that should handle this grievance from the following list: "
                        "Water-department, Electricity-department, Roads-department, Traffic-department, Fire-department, Police-department, "
                        "Health-department, Education-department, Agriculture-department, Other.\n\n"
                        "Important:\n"
                        "- Select the department based on who can solve the issue, not just matching the photo.\n"
                        "- Output should be in strict JSON format like:\n"
                        "{\n"
                        "  \"title\": \"...\",\n"
                        "  \"description\": \"...\",\n"
                        "  \"category\": \"...\"\n"
                        "}"
                    )
                }
            ]
        }]
    }

    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code != 200:
            return jsonify({"error": "Failed to process image", "details": response.json()}), response.status_code

        return jsonify(response.json())

    except Exception as e:
        return jsonify({"error": "An error occurred", "details": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
