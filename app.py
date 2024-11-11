import streamlit as st
import numpy as np
import requests
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from PIL import Image

# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet')

# Function to get additional information about the highest prediction
def get_info_about_object(label):
    try:
        response = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{label}")
        data = response.json()
        
        if response.status_code == 200 and 'extract' in data:
            summary = data['extract']
            return summary, data.get('content_urls', {}).get('desktop', {}).get('page', None)
        else:
            return "No additional information available.", None
    except Exception as e:
        return f"Error fetching data: {str(e)}", None

# Function to compute Grad-CAM
def compute_gradcam(model, img_array, class_idx):
    grad_model = Model(inputs=[model.inputs],
                       outputs=[model.get_layer("block5_conv3").output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

# Function to overlay the heatmap on the image
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
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image for VGG16
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions using the VGG16 model
    predictions = model.predict(img_array)

    # Decode and display the predictions
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    st.write("Top Predictions:")
    
    # Extract labels and scores
    labels, scores = [], []
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        labels.append(label)
        scores.append(score)
        st.write(f"{i+1}. {label}: {round(score*100, 2)}% confidence")

    # Get highest prediction (top one)
    top_label = decoded_predictions[0][1]
    top_class_idx = np.argmax(predictions[0])
    st.write(f"\n### Highest Prediction: {top_label}")

    # Fetch and display additional information about the top prediction
    summary, wiki_url = get_info_about_object(top_label)
    st.write(f"**More Info:** {summary}")
    if wiki_url:
        st.write(f"[Read more about {top_label} on Wikipedia]({wiki_url})")

    # Compute and display Grad-CAM
    st.write("Generating Grad-CAM visualization...")
    heatmap = compute_gradcam(model, img_array, top_class_idx)
    overlayed_img = overlay_gradcam(img, heatmap)
    st.image(overlayed_img, caption='Grad-CAM Heatmap', use_column_width=True)

    # New Feature: Confidence Score Visualization
    st.write("### Confidence Score Visualization")
    plt.figure(figsize=(8, 4))
    plt.bar(labels, scores, color='skyblue')
    plt.ylim(0, 1)
    plt.xlabel('Classes')
    plt.ylabel('Confidence Score')
    plt.title('Confidence Scores for Top Predictions')
    st.pyplot(plt)
