import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your trained model
model = tf.keras.models.load_model("best_teeth_model1.h5", compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Class labels
class_names = ["CaS", "CoS", "Gum", "MC", "OC", "OLP", "OT"]

# Same preprocessing as image_dataset_from_directory
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    image = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)  # keep scale 0–255
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit UI
st.title("🦷 Oral Illness Classifier")
st.write("Upload an oral image to classify it into one of 7 disease classes.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        input_tensor = preprocess_image(image)

        # Predict
        prediction = model.predict(input_tensor)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100

        st.success(f"🩺 Predicted Illness: **{predicted_class}**")
        st.info(f"📈 Confidence: **{confidence:.2f}%**")
