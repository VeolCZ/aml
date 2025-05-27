import os
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import base64
import requests
import json
import sys
from draw_bounding_box import draw_bounding_box

sys.path.append(os.getcwd())


def encode_image_to_base64(image: bytes) -> str:
    """
    Encode an image in bytes representation into a string

    Args:
        image: the image to encode

    Returns:
        str: the encoded image
    """
    encoded_string = base64.b64encode(image).decode("utf-8")
    return encoded_string


def showcase_prediction_results(results: dict, image: str | UploadedFile) -> None:
    """
    Showcase the results nicely in the streamlit app

    Args:
        results: dictionary containing the results
        image: the image on which to draw the bounding box
    """
    st.markdown(f"**:blue[Specie]**: {results['species']}")
    st.markdown(f"**:red[Bounding Box]**: {results['bounding_box']}")
    st.image(draw_bounding_box(image, results["bounding_box"]))


st.title("Bird Classification and Bounding Box Estimation")

model_selection = st.selectbox(
    "Pick your model",
    options=["Visual Transformer", "Random Forest"],
    placeholder="Select model",
)

if model_selection == "Visual Transformer":
    model = "ViT"
elif model_selection == "Random Forest":
    model = "Forest"

select_preselected_image = st.toggle("Do you want to try out a preselected image?")

if not select_preselected_image:
    image_selection = st.file_uploader("Input your image", type=["jpg", "jpeg", "png"])
    if image_selection is not None:
        image = base64.b64encode(image_selection.read()).decode("utf-8")
else:
    path = "/aml/streamlit_app/preselected_images"
    image_selection = st.selectbox(
        "Choose a preselected image",
        os.listdir(path),
    )
    image_selection = path + "/" + image_selection
    with open(image_selection, "rb") as image_file:
        image = base64.b64encode(image_file.read()).decode("utf-8")

if image_selection:

    st.image(image_selection)

    predict = st.button("PREDICT!")

    if predict:
        payload = {"image": image, "model": model}
        headers = {"Content-Type": "application/json"}
        prediction = requests.post(
            "http://localhost:8000/predict", headers=headers, data=json.dumps(payload)
        )
        showcase_prediction_results(prediction.json(), image_selection)
