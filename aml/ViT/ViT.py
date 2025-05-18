import torch
import os
from transformers import ViTForImageClassification
from sklearn.metrics import top_k_accuracy_score
from interface.ModelInterface import ModelInterface
from datetime import datetime
from ViT.ViTTrainer import ViTTrainer
from preprocessing.ViTImageDataset import ViTImageDataset


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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        SEED = int(os.getenv("SEED", 123))
        torch.manual_seed(SEED)
        batch_size = 350
        model_path = f"/data/ViT_{datetime.utcnow()}"
        learning_rate = 0.0012278101209126883
        annealing_rate = 6.1313110341652e-07
        n_of_folds = 10
        epochs = 20
        patience = 4

        trainer = ViTTrainer(self, device, dataset,
                             epochs=epochs, batch_size=batch_size, patience=patience,
                             learning_rate=learning_rate, n_splits=n_of_folds, annealing_rate=annealing_rate)
        trainer.train(model_path=model_path, save=True)

    def predict(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device=device)
        data.to(device=device)

        self.eval()
        bbox: torch.Tensor
        cls: torch.Tensor
        with torch.no_grad():
            bbox, cls = self(data)

        return bbox, cls

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, weights_only=True))

    def evaluate(self, predictions: tuple[torch.Tensor, torch.Tensor],
                 label: tuple[torch.Tensor, torch.Tensor]) -> float:
        """
        Evaluates the model using Intersection over union, top k accuracy.
        ARGS:
        Predictions(Tuple[torch.Tensor, torch.Tensor]): A tuple containing two tensors:
            - bbox_preds (torch.Tensor): The predicted bounding boxes from bbox_head.
                                         Shape: (batch_size, 4).
            - cls_preds (torch.Tensor): The predicted classification logits from cls_head.
                                        Shape: (batch_size, num_classes).
        label(Tuple[torch.Tensor, torch.Tensor]): A tuple containing two tensors:
            - bbox_label (torch.Tensor): The labeled bounding boxes.
                                         Shape: (batch_size, 4). This batch_size should be
                                         the same as batchsize of predictions.
            - cls_label (torch.Tensor): The labeled bounding boxes.
                                        Shape: (batch_size). This batch_size should be the
                                        same as batchsize of predictions.
        Returns
        metric_outcomes(tuple[floats]): tuple in order (multiple IoU, Top1 accuracy, top 5 accuracy, top 10 accuracy)
        """
        cls_label = [x for x in range(1, 201)]

        bbox_pred = predictions[0]
        class_pred = predictions[1]
        bbox_label = label[0]

        top1_accuracy = top_k_accuracy_score(label[1], class_pred, k=1, label=cls_label)
        top5_accuracy = top_k_accuracy_score(label[1], class_pred, k=5, label=cls_label)
        top10_accuracy = top_k_accuracy_score(label[1], class_pred, k=10, label=cls_label)

        iou_sum = 0
        for batchnum in range(bbox_pred.size[0]):
            iou_sum += self.compoute_iou(bbox_pred[batchnum], bbox_label[batchnum])

        multiple_iou = iou_sum / bbox_pred.size[0]

        return multiple_iou, top1_accuracy, top5_accuracy, top10_accuracy

    def compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        """
        Computes the Intersection over Union (IoU) of two bounding boxes.
        Args:
            box1 (torch.Tensor): The first bounding box.
            box2 (torch.Tensor): The second bounding box.
        Returns:
            float: The IoU score.
        """
        x1 = torch.max(torch.tensor([box1[0], box2[0]]))
        y1 = torch.max(torch.tensor([box1[1], box2[1]]))
        x2 = torch.min(torch.tensor([box1[2], box2[2]]))
        y2 = torch.min(torch.tensor([box1[3], box2[3]]))

        inter_area = torch.max(torch.tensor(0), x2 - x1) * torch.max(torch.tensor(0), y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        return float((inter_area / union_area).item()) if union_area > 0 else 0
