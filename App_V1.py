import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import io

# ---------------------------------
# 1. Configuration
# ---------------------------------
class_names = ["CaS", "CoS", "Gum", "MC", "OC", "OLP", "OT"]
img_size = (224, 224)

# ---------------------------------
# 2. Load Model
# ---------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mobilenet_model.h5", compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_model()

# ---------------------------------
# 3. Preprocess Function
# ---------------------------------
def preprocess_image(image: Image.Image):
    image = image.resize(img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.cast(img_array, tf.float32) / 255.0  # Normalize
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array

# ---------------------------------
# 4. UI
# ---------------------------------
st.set_page_config(page_title="Oral Illness Classifier", layout="centered")
st.title("🦷 Oral Illness Classifier")
st.markdown("Upload an oral image to classify it into one of the 7 disease categories.")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        input_tensor = preprocess_image(image)
        prediction = model.predict(input_tensor)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100

        st.success(f"🩺 **Predicted Illness:** {predicted_class}")
        st.info(f"📈 **Confidence:** {confidence:.2f}%")

# ---------------------------------
# 5. Evaluation Section
# ---------------------------------
st.markdown("---")
st.subheader("📊 Evaluate Model on Test Set")

test_eval = st.checkbox("Run full evaluation (confusion matrix, classification report)")

if test_eval:
    with st.spinner("Loading and evaluating test set..."):

        test_dir = "D:\projects\Teeth Classification\Teeth_Dataset\Testing"  # Change if needed
        batch_size = 32
        AUTOTUNE = tf.data.AUTOTUNE

        def preprocess(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            return image, label

        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            test_dir,
            labels="inferred",
            label_mode="categorical",
            image_size=img_size,
            shuffle=False,
            batch_size=batch_size
        ).map(preprocess).prefetch(buffer_size=AUTOTUNE)

        # Evaluate
        loss, accuracy = model.evaluate(test_ds, verbose=0)
        st.success(f"✅ **Test Accuracy:** {accuracy * 100:.2f}%")

        # Predict and Collect True/Pred Labels
        y_pred_probs = model.predict(test_ds)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.concatenate([np.argmax(label.numpy(), axis=1) for _, label in test_ds])

        # Classification Report
        st.subheader("📄 Classification Report")
        report_str = classification_report(y_true, y_pred, target_names=class_names)
        st.code(report_str, language="text")

        # Confusion Matrix
        st.subheader("🔢 Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        st.pyplot(fig)
