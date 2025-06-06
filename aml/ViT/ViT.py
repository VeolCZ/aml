import multiprocessing
from preprocessing.ViTImageDataset import ViTImageDataset
import torch
import os
from transformers import ViTForImageClassification
from interface.ModelInterface import ModelInterface
from torch.utils.data import DataLoader

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ViT(torch.nn.Module, ModelInterface):
    def __init__(self, hidden_size: int = 1024, num_classes: int = 200, dp_rate: float = 0.05) -> None:
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
        num_encoder_layers_to_unfreeze = 3

        self.backbone = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224", cache_dir="/data/vit")
        self.backbone.classifier = torch.nn.Identity()

        for param in self.backbone.parameters():
            param.requires_grad = False

        encoder_layers = self.backbone.vit.encoder.layer
        total_encoder_layers = len(encoder_layers)

        layers_to_unfreeze = min(num_encoder_layers_to_unfreeze, total_encoder_layers)

        for i in range(total_encoder_layers - layers_to_unfreeze, total_encoder_layers):
            for param in encoder_layers[i].parameters():
                param.requires_grad = True

        for param in self.backbone.vit.layernorm.parameters():
            param.requires_grad = True

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
            torch.nn.Sigmoid(),
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

    def fit(self, train_dataset: ViTImageDataset, val_dataset: ViTImageDataset) -> None:
        from ViT.ViTTrainer import ViTTrainer  # Import as needed

        learning_rate = 4.5044719925484676e-04
        annealing_rate = 2.9188464128352595e-06
        epochs = 15
        patience = 4

        trainer = ViTTrainer(self, DEVICE,
                             epochs=epochs, batch_size=BATCH_SIZE, patience=patience,
                             learning_rate=learning_rate, annealing_rate=annealing_rate)
        traind_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=multiprocessing.cpu_count(), pin_memory=DEVICE.type == "cuda")
        val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=multiprocessing.cpu_count(), pin_memory=DEVICE.type == "cuda")
        trainer.train(traind_dl, val_dl)

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

    def save(self, path: str) -> None:
        """
        Saves both random forests

        Args:
            path(str): the path to the directory where the forests need to be saved.
        """
        torch.save(self.state_dict(), path + ".pth")
