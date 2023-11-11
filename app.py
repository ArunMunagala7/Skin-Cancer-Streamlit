# Import necessary libraries
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('cnn_fc_model_1.h5')

# Define a function to make predictions
def predict(image):
    # Preprocess the image
    img = tf.keras.preprocessing.image.load_img(image, target_size=(150, 150,3))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    predictions = model.predict(img_array)
    return predictions

# Streamlit app
st.title("Simple Model Deployment with Streamlit")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Make predictions when the user clicks the "Predict" button
    if st.button("Predict"):
        predictions = predict(uploaded_file)
        class_names = ["actinic keratosis", "basal cell carcinoma", "dermatofibroma","melanoma","nevus","pigmented benign keratosis","seborrheic keratosis","squamous cell carcinoma","vascular lesion"]  # Replace with your class names

        # Display the predicted class and confidence
        st.write(f"Prediction: {class_names[np.argmax(predictions)]}")
        st.write(f"Confidence: {100 * np.max(predictions):.2f}%")
