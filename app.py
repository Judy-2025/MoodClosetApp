import streamlit as st
import numpy as np
from PIL import Image
import cv2
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load Model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("MoodClosetModel.h5")
    return model

model = load_model()

# Mood → Colors
mood_to_colors = {
    "Happy 😊": ["yellow", "orange", "bright red"],
    "Calm 😌": ["pastel blue", "mint", "lavender"],
    "Confident 😎": ["black", "navy", "bold red"],
    "Relaxed 😴": ["white", "beige", "soft green"]
}

# Skin tone → Colors
skin_tone_colors = {
    "warm": ["yellow", "orange", "red"],
    "cool": ["blue", "purple", "mint"],
    "neutral": ["white", "beige", "gray"]
}

def classify_skin_tone(image):
    img = np.array(image)
    avg = img.mean(axis=(0, 1))  # RGB

    # Simple rule
    r, g, b = avg
    if r > b:
        return "warm"
    elif b > r:
        return "cool"
    else:
        return "neutral"

def detect_dominant_colors(image):
    img = np.array(image)
    img = cv2.resize(img, (64, 64))
    img = img.reshape((-1, 3))

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(img)
    return kmeans.cluster_centers_.astype(int)

def predict_clothing(image):
    img = image.resize((128, 128))
    img = np.expand_dims(np.array(img), axis=0)
    img = preprocess_input(img)

    preds = model.predict(img)
    class_index = np.argmax(preds)
    return class_index

# Streamlit UI
st.title("👗 MoodCloset – AI Fashion Recommender")
st.write("Upload your clothing item + selfie to receive personalized recommendations.")

# --- CLOTHING UPLOAD ---
clothes_file = st.file_uploader("Upload a clothing image", type=["jpg", "png", "jpeg"])

if clothes_file:
    clothing_image = Image.open(clothes_file).convert("RGB")
    st.image(clothing_image, caption="Your Clothing Item", width=300)

    class_index = predict_clothing(clothing_image)
    st.success(f"Detected Clothing Category: **{list(range(10))[class_index]}** (placeholder)")

    colors = detect_dominant_colors(clothing_image)
    st.write("Dominant Clothing Colors:", colors)

# --- SELFIE UPLOAD ---
selfie_file = st.file_uploader("Upload your selfie", type=["jpg", "png", "jpeg"])

if selfie_file:
    selfie_image = Image.open(selfie_file).convert("RGB")
    st.image(selfie_image, caption="Your Selfie", width=300)

    undertone = classify_skin_tone(selfie_image)
    st.success(f"Detected Skin Undertone: **{undertone}**")

# --- MOOD SELECTION ---
mood = st.selectbox("Select your current mood", list(mood_to_colors.keys()))

# --- RECOMMENDATION BUTTON ---
if st.button("Get outfit recommendation"):
    if not clothes_file:
        st.error("Upload a clothing image first.")
    elif not selfie_file:
        st.error("Upload a selfie first.")
    else:
        st.subheader("🎉 Your Outfit Recommendation")

        recommended = list(
            set(mood_to_colors[mood])
            & set(skin_tone_colors[undertone])
        )

        if not recommended:
            recommended = mood_to_colors[mood]

        st.write(f"Recommended colors for your mood **{mood}** and undertone **{undertone}**:")
        st.success(recommended)
