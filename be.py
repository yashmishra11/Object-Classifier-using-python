import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from tensorflow.keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import logging

# Initialize Flask app and logging
app = Flask(__name__)
CORS(app)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

logging.basicConfig(level=logging.DEBUG)

# Load the pre-trained VGG16 model
model = VGG16(weights="imagenet")

def generate_grad_cam(model, img_array, class_index):
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer("block5_conv3").output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)[0]
        guided_grads = tf.maximum(grads, 0)
        weights = tf.reduce_mean(guided_grads, axis=(0, 1))

        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1).numpy()

        # Normalize and resize the Grad-CAM heatmap
        cam = np.maximum(cam, 0)
        cam = cam / cam.max() if cam.max() != 0 else cam
        cam = cv2.resize(cam, (224, 224))
        return cam
    except Exception as e:
        logging.error(f"Error in generating Grad-CAM: {str(e)}")
        raise

def generate_prediction_graph(predictions):
    try:
        labels = [label for _, label, _ in predictions]
        scores = [float(score) for _, _, score in predictions]

        # Create a bar chart
        fig, ax = plt.subplots()
        ax.barh(labels, scores, color='skyblue')
        ax.set_xlabel('Confidence')
        ax.set_title('Top Predictions')

        # Save the plot to a BytesIO object and convert to base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_b64 = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)  # Close the plot to avoid resource locks

        return img_b64
    except Exception as e:
        logging.error(f"Error in generating prediction graph: {str(e)}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the image file is in the request
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided."}), 400

        file = request.files['image']

        # Validate file
        if file.filename == '':
            return jsonify({"error": "No selected file."}), 400
        if not file.filename.lower().endswith(('jpg', 'jpeg', 'png')):
            return jsonify({"error": "Invalid file type. Please upload a jpg, jpeg, or png file."}), 400

        # Open and preprocess the image
        img = Image.open(file.stream)
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make predictions
        predictions = model.predict(img_array)
        if predictions is None or len(predictions) == 0:
            return jsonify({"error": "No predictions were made by VGG16."}), 400

        decoded_predictions = decode_predictions(predictions, top=3)[0]
        results = [{"label": label, "confidence": float(score)} for (_, label, score) in decoded_predictions]

        # Generate prediction graph
        graph_b64 = generate_prediction_graph(decoded_predictions)

        # Get the top class for Grad-CAM
        top_class = np.argmax(predictions[0])

        # Generate Grad-CAM heatmap
        cam = generate_grad_cam(model, img_array, top_class)

        # Overlay heatmap on the image
        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay_img = cv2.addWeighted(cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)

        # Ensure the static folder exists
        if not os.path.exists('static'):
            os.makedirs('static')

        # Save Grad-CAM overlay
        grad_cam_path = "static/grad_cam.jpg"
        cv2.imwrite(grad_cam_path, overlay_img)

        # Return the results
        return jsonify({
            "predictions": results,
            "source": "VGG16",
            "grad_cam_path": grad_cam_path,
            "prediction_graph": graph_b64
        })

    except Exception as e:
        logging.error(f"An error occurred in /predict: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
