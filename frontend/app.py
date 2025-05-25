import streamlit as st
import requests

st.title("Fashion MNIST Classifier")

uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file:
    files = {"file": uploaded_file.getvalue()}
    response = requests.post("http://localhost:8000/predict", files=files)

    if response.ok:
        data = response.json()
        st.image(uploaded_file, caption="Uploaded Image")
        st.write(f"**Predicted Class:** {data['class']}")
        st.write(f"**Confidence:** {data['confidence']:.4f}")
    else:
        st.error("Prediction failed. Try another image.")
