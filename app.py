import streamlit as st
import numpy as np
from PIL import Image
import pickle
import time

st.title("Image Classification Chatbot")
st.write("Upload an image, and the model will predict its class in a chatbot-style")

# Load the model from the pickle file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# CIFAR-10 class names
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Sidebar: Display prediction history as a chat conversation
if "history" not in st.session_state:
    st.session_state.history = []

st.sidebar.header("Prediction History")
for record in st.session_state.history:
    st.sidebar.write(f"{record['class']} - {record['time']:.2f}s")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((32, 32))
    
    # Display the chatbot conversation history
    st.write("### Chatbot Conversation")
    for record in st.session_state.history:
        col1, col2 = st.columns([1, 3], gap="large")  # Adjusted for better spacing
        with col1:
            st.image(record['image'], width=150)  # Adjusted width to prevent overlap
        with col2:
            st.write(f"Prediction: **{record['class']}**")
            st.write(f"Predicted time: {record['time']:.2f} seconds")
        st.divider()
    
    # Normalize and expand dimensions for model input
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Measure prediction time
    start_time = time.time()
    with st.spinner("Classifying..."):
        predictions = model.predict(image_array)
    end_time = time.time()
    
    # Calculate elapsed time and show prediction result
    elapsed_time = end_time - start_time
    predicted_class = class_names[np.argmax(predictions)]
    
    # Update history with the image and prediction result
    st.session_state.history.append({
        "image": image,
        "class": predicted_class,
        "time": elapsed_time
    })
    
    # Display the latest uploaded image and prediction in chatbot style
    col1, col2 = st.columns([1, 3], gap="large")  # Adjusted for better spacing
    with col1:
        st.image(image, width=150)  # Adjusted width to prevent overlap
    with col2:
        st.write(f"Prediction: **{predicted_class}**")
        st.write(f"Predicted time: {elapsed_time:.2f} seconds")







    
    



