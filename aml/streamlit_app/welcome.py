import os
import streamlit as st
import base64
import requests
import json
import sys

sys.path.append(os.getcwd())


def encode_image_to_base64(image) -> str:
    encoded_string = base64.b64encode(image).decode("utf-8")
    return encoded_string


st.title("Bird classification and bounding box estimation")

model_selection = st.selectbox(
    "Pick you model",
    options=["Visual Transformer", "Random Forest"],
    placeholder="Select model",
)

if model_selection == "Visual Transformer":
    model = "ViT"
elif model_selection == "Random Forest":
    model = "Forest"

select_preselected_image = st.toggle("Do you want to try out a preselected image?")

if not select_preselected_image:
    image = st.file_uploader("Input you image", type=["jpg", "jpeg", "png"])
else:
    path = "/aml/streamlit_app/preselected_images"
    image_selection = st.selectbox(
        "Choose a preselected image",
        os.listdir(path),
    )
    image_selection = path + "/" + image_selection
    with open(image_selection, "rb") as image_file:
        image = base64.b64encode(image_file.read()).decode("utf-8")

if image is not None and model is not None:

    predict = st.button("PREDICT!")

    if predict:
        payload = {"image": encode_image_to_base64(image), "model": model}
        headers = {"Content-Type": "application/json"}
        prediction = requests.post(
            "http://localhost:8000/predict", headers=headers, data=json.dumps(payload)
        )
        st.write(prediction)
