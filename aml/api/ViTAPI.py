from fastapi import HTTPException, status
from PIL import Image, UnidentifiedImageError
from preprocessing.ViTPreprocessPipeline import ViTPreprocessPipeline
from pydantic import BaseModel, Field
from typing import Any
from ViT.ViT import ViT
import base64
import io
import litserve as ls
import numpy as np
import torch


class APIInput(BaseModel):
    """
    Schema for image prediction request.
    """
    image: str = Field(
        ...,
        description=(
            "Base64 encoded string of the input image (JPEG or PNG recommended). "
        )
    )


class APIOutput(BaseModel):
    """
    Schema for image prediction response.
    """
    class_id: int = Field(
        ...,
        description="The predicted class ID (integer)."
    )
    bounding_box: list[float] = Field(
        ...,
        description="Predicted bounding box coordinates [x_min, y_min, x_max, y_max].",
        min_length=4,
        max_length=4
    )


class ViTAPI(ls.LitAPI):
    """
    LitServe API for image classification and bounding box prediction using a Vision Transformer.
    """

    def setup(self, device: str) -> None:
        """
        Initializes the ViT model and preprocessing pipeline.

        Args:
            device (str): Compute device to use ("cuda" or "cpu").
        """
        self.device = device
        self.vit = ViT()
        self.vit.load("/weights/ViT_2025-05-16 12:28:42.294240ValLoss_1.84.pth")
        self.preprocess = ViTPreprocessPipeline.get_base_eval_transform()

    def decode_request(self, request: APIInput) -> torch.Tensor:
        """
        Decodes base64 image from request into a preprocessed PyTorch tensor.

        Args:
            request (APIInput): Incoming API request containing the base64 image.

        Returns:
            torch.Tensor: Preprocessed image tensor ready for the model.

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
        image_tensor: torch.Tensor = self.preprocess(
            image=image_np,
            bboxes=[],
            class_labels=[]
        )["image"]

        return image_tensor.unsqueeze(0)

    def encode_response(self, output: dict[str, Any]) -> APIOutput:
        """
        Encodes model"s raw prediction output into the APIOutput schema.

        Args:
            output (dict[str, Any]): Raw prediction results from predict.

        Returns:
            APIOutput: Formatted API response.
        """
        return APIOutput(**output)

    def predict(self, data: torch.Tensor) -> dict[str, Any]:
        """
        Performs inference using the ViT model.

        Args:
            data (torch.Tensor): Preprocessed image tensor.

        Returns:
            dict[str, Any]: Contains "class_id" (int) and "bounding_box" (list of floats).
        """
        input_tensor = data.to(self.device)
        bbox, cls = self.vit.predict(input_tensor)

        bbox_list = bbox.squeeze(0).cpu().numpy().tolist()
        class_id = cls.argmax(-1).item()

        return {"class_id": class_id, "bounding_box": bbox_list}


def serve() -> None:
    """
    Starts the LitServe API server.
    """
    server = ls.LitServer(ViTAPI(max_batch_size=1), accelerator="auto")
    server.run(port=8000, host="0.0.0.0")
