import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from PIL import Image

# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet')

# Grad-CAM function
def compute_gradcam(model, img_array, class_idx):
    try:
        grad_model = Model(inputs=[model.inputs], outputs=[model.get_layer("block5_conv3").output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
        return heatmap.numpy()
    except Exception as e:
        st.error(f"Error computing Grad-CAM: {str(e)}")
        return None

def overlay_gradcam(img, heatmap, alpha=0.6, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    img = np.array(img)
    overlayed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return overlayed_img

# Streamlit App
st.title("Object Identification with Grad-CAM using VGG16")
st.write("Upload an image or take a photo with the camera. The model will predict the object in the image and visualize important regions using Grad-CAM.")

# Input options
option = st.radio("Choose input method:", ('Upload Image', 'Take Photo with Camera'))

if option == 'Upload Image':
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("Take a photo")

if uploaded_file is not None:
    # Open the image
    img = Image.open(uploaded_file)

    # Ensure image is in RGB mode
    if img.mode != 'RGB':
        img = img.convert('RGB')

    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image for VGG16
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions using the VGG16 model
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    st.write("### VGG16 Predictions:")
    vgg_predictions = []
    for i, (_, label, score) in enumerate(decoded_predictions):
        vgg_predictions.append({"label": label, "confidence": score})
        st.write(f"{i+1}. {label}: {round(score * 100, 2)}% confidence")

    # Grad-CAM Visualization
    try:
        top_class_idx = np.argmax(predictions[0])
        heatmap = compute_gradcam(model, img_array, top_class_idx)

        if heatmap is not None:
            overlayed_img = overlay_gradcam(img, heatmap)
            st.image(overlayed_img, caption='Grad-CAM Heatmap', use_column_width=True)
        else:
            st.error("Failed to generate Grad-CAM heatmap.")
    except Exception as e:
        st.error(f"Error generating Grad-CAM: {str(e)}")
