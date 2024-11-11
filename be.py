from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from tensorflow.keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
from PIL import Image

app = Flask(__name__)
CORS(app) 

# Load the pre-trained VGG16 model
model = VGG16(weights="imagenet")

# Function to get additional information about the highest prediction
def get_info_about_object(label):
    try:
        response = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{label}")
        if response.status_code == 200:
            data = response.json()
            return data.get("extract", "No info available"), data.get("content_urls", {}).get("desktop", {}).get("page", None)
        return "No additional information available.", None
    except Exception as e:
        return f"Error fetching data: {str(e)}", None

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(file.stream)
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    results = [
        {"label": label, "confidence": float(score)}
        for (_, label, score) in decoded_predictions
    ]

    top_label = decoded_predictions[0][1]
    summary, wiki_url = get_info_about_object(top_label)

    return jsonify({
        "predictions": results,
        "top_label": top_label,
        "summary": summary,
        "wiki_url": wiki_url
    })

if __name__ == "__main__":
    app.run(debug=True)
