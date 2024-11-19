from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from tensorflow.keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf

app = Flask(__name__)
CORS(app)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load the pre-trained VGG16 model
model = VGG16(weights="imagenet")

def generate_grad_cam(model, img_array, class_index):
    """
    Generate Grad-CAM heatmap for a given image tensor and model.
    """
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
    cam = cam / cam.max()
    cam = cv2.resize(cam, (224, 224))
    return cam


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the image file is in the request
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided."}), 400

        file = request.files['image']

        # Ensure the file has a valid name and type
        if file.filename == '':
            return jsonify({"error": "No selected file."}), 400
        if not file.filename.lower().endswith(('jpg', 'jpeg', 'png')):
            return jsonify({"error": "Invalid file type. Please upload a jpg, jpeg, or png file."}), 400

        # Open and save the image temporarily
        img = Image.open(file.stream)
        img_path = "temp_image.jpg"
        img.save(img_path)

        # Preprocess the image for VGG16
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make predictions using VGG16
        predictions = model.predict(img_array)
        if predictions is None or len(predictions) == 0:
            return jsonify({"error": "No predictions were made by VGG16."}), 400

        # Decode VGG16 predictions
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        results = [{"label": label, "confidence": float(score)} for (_, label, score) in decoded_predictions]

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

        # Return the results from VGG16 and Grad-CAM
        return jsonify({
            "predictions": results,
            "source": "VGG16",
            "grad_cam_path": grad_cam_path
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
