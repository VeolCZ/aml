from enum import Enum
from fastapi import HTTPException, status
from PIL import Image, UnidentifiedImageError
from random_forests.CompositeRandomForest import CompositeRandomForest
from preprocessing.ViTPreprocessPipeline import ViTPreprocessPipeline
from pydantic import BaseModel, Field
from typing import Any
from ViT.ViT import ViT
import base64
import io
import litserve as ls
import numpy as np
import torch
from preprocessing.TreePrerocessPipeline import TreePrerocessPipeline


class ModelType(str, Enum):
    """Enumeration for available model types."""
    VIT = "ViT"
    Forest = "Forest"


class APIInput(BaseModel):
    """Schema for image prediction request."""
    image: str = Field(
        ...,
        description=(
            "Base64 encoded string of the input image (JPEG or PNG recommended)."
        )
    )
    model: ModelType = Field(
        ...,
        description="Model to use for prediction (Forest/ViT)."
    )


class APIOutput(BaseModel):
    """Schema for image prediction response."""
    class_id: int = Field(
        ...,
        description="The predicted class ID (integer)."
    )
    bounding_box: list[float] = Field(
        ...,
        description="Predicted bounding box coordinates [x_min, y_min, x_max, y_max] scaled to [0-1] range.",
        min_length=4,
        max_length=4
    )


class ViTAPI(ls.LitAPI):
    """
    LitServe API for image classification and bounding box prediction.

    Uses either a Vision Transformer (ViT) or a Random Forest model.
    """

    def setup(self, device: str) -> None:
        """
        Initializes models and preprocessing pipelines.

        Args:
            device (str): Compute device to use ("cuda" or "cpu").
        """
        self.device = device
        self.vit = ViT()
        self.vit.load("/weights/ViT_2025-05-16 12:28:42.294240ValLoss_1.84.pth")
        self.forest = CompositeRandomForest()
        self.forest.load("/weights/forest")
        self.vit_preprocess = ViTPreprocessPipeline.vit_predict_transform
        self.tree_preprocess = TreePrerocessPipeline.tree_predict_transform

    def decode_request(self, request: APIInput) -> tuple[torch.Tensor, ModelType]:
        """
        Decodes base64 image from request into a preprocessed PyTorch tensor.

        Args:
            request (APIInput): Incoming API request containing the base64 image.

        Returns:
            tuple[torch.Tensor, ModelType]: Preprocessed image tensor and the selected model type.

        Raises:
            HTTPException: If base64 string is invalid, image is malformed, or preprocessing fails.
        """
        image_b64 = request.image
        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid base64 string. Ensure the image is correctly encoded."
            )

        try:
            image_file = Image.open(io.BytesIO(image_bytes))
            image_file.verify()
            image_file = Image.open(io.BytesIO(image_bytes))
            if image_file.mode != "RGB":
                image = image_file.convert("RGB")
            else:
                image = image_file
        except (UnidentifiedImageError, OSError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Malformed image data or unsupported image format. Error: {e}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"An unexpected error occurred while processing the image: {e}"
            )

        image_np = np.array(image)
        if request.model == ModelType.VIT:
            image_tensor = self.vit_preprocess(image_np)
        else:
            image_tensor = self.tree_preprocess(image_np)

        return image_tensor, request.model

    def encode_response(self, output: dict[str, Any]) -> APIOutput:
        """
        Encodes raw prediction output into the APIOutput schema.

        Args:
            output (dict[str, Any]): Raw prediction results.

        Returns:
            APIOutput: Formatted API response.
        """
        return APIOutput(**output)

    def predict(self, data: tuple[torch.Tensor, ModelType]) -> dict[str, Any]:
        """
        Performs inference using the selected model.

        Args:
            data (tuple[torch.Tensor, ModelType]): Preprocessed image tensor and model type.

        Returns:
            dict[str, Any]: Contains "class_id" (int) and "bounding_box" (list of floats).
        """
        input_tensor = data[0].to(self.device)
        if data[1] == ModelType.VIT:
            bbox, cls = self.vit.predict(input_tensor)
            bbox = bbox / ViTPreprocessPipeline.img_size
        else:
            bbox, cls = self.forest.predict(input_tensor)
            bbox = bbox / TreePrerocessPipeline.img_size

        bbox_list = bbox.squeeze(0).cpu().numpy().tolist()
        class_id = cls.argmax(-1).item()

        return {"class_id": class_id, "bounding_box": bbox_list}


def serve() -> None:
    """Starts the LitServe API server."""
    server = ls.LitServer(ViTAPI(max_batch_size=1), accelerator="auto")
    server.run(port=8000, host="0.0.0.0")
