from preprocessing.ViTImageDataset import ViTImageDataset
import torch
import os
from transformers import ViTForImageClassification
from interface.ModelInterface import ModelInterface
from datetime import datetime

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ViT(torch.nn.Module, ModelInterface):
    def __init__(self, hidden_size: int = 1024, num_classes: int = 200, dp_rate: float = 0.1) -> None:
        """
        Initializes the ViT model with a pre-trained backbone and custom heads.

        Args:
            hidden_size (int, optional): The size of the hidden layers in the
                                         classification and bounding box heads.
                                         Defaults to 1024.
            num_classes (int, optional): The number of output classes for the
                                         classification head. Defaults to 200.
            dp_rate (float, optional): The dropout rate for the heads. Defaults to 0.1.
        """
        super(ViT, self).__init__()
        backbone_out_size = 768

        self.backbone = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224", cache_dir="/data/vit")
        self.backbone.classifier = torch.nn.Identity()

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(backbone_out_size, hidden_size),
            torch.nn.Dropout(dp_rate),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.Dropout(dp_rate),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size // 2, hidden_size // 4),
            torch.nn.Dropout(dp_rate),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size // 4, num_classes),
        )

        self.bbox_head = torch.nn.Sequential(
            torch.nn.Linear(backbone_out_size, hidden_size),
            torch.nn.Dropout(dp_rate),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.Dropout(dp_rate),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size // 2, hidden_size // 4),
            torch.nn.Dropout(dp_rate),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size // 4, 4),
        )

        for layer in self.cls_head + self.bbox_head:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): The input image tensor. Expected shape is
                              (batch_size, num_channels, height, width).
                              The model is pre-trained on 224x224 images, so inputs
                              should ideally be of compatible size or handled by
                              preprocessing before being passed here.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
                - bbox_preds (torch.Tensor): The predicted bounding boxes from bbox_head.
                                             Shape: (batch_size, 4).
                - cls_preds (torch.Tensor): The predicted classification logits from cls_head.
                                            Shape: (batch_size, num_classes).
        """
        backbone = self.backbone(pixel_values=x).logits
        bbox = self.bbox_head(backbone)
        cls = self.cls_head(backbone)
        return bbox, cls

    def fit(self, dataset: ViTImageDataset) -> None:
        from ViT.ViTTrainer import ViTTrainer  # Import as needed

        model_path = f"/data/ViT_{datetime.utcnow()}"
        learning_rate = 0.0012278101209126883
        annealing_rate = 6.1313110341652e-07
        n_of_folds = 10
        epochs = 20
        patience = 4

        trainer = ViTTrainer(self, DEVICE, dataset,
                             epochs=epochs, batch_size=BATCH_SIZE, patience=patience,
                             learning_rate=learning_rate, n_splits=n_of_folds, annealing_rate=annealing_rate)
        trainer.train(model_path=model_path, save=True)

    @torch.inference_mode()
    def predict(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.to(device=DEVICE)
        self.eval()

        all_bbox = []
        all_cls = []
        num_batches = (data.shape[0] + BATCH_SIZE - 1) // BATCH_SIZE

        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, data.shape[0])
            batch_data = data[start_idx:end_idx].to(device=DEVICE)
            bbox_batch, cls_batch = self(batch_data)

            all_bbox.append(bbox_batch)
            all_cls.append(torch.nn.functional.softmax(cls_batch, dim=-1))

        final_bbox = torch.cat(all_bbox, dim=0).to("cpu")
        final_cls = torch.cat(all_cls, dim=0).to("cpu")

        return final_bbox, final_cls

    def load(self, path: str) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device=device)
        self.load_state_dict(torch.load(path, map_location=device))
