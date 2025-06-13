import logging
import os
import torch
import torchvision
from ViT.ViT import ViT
from preprocessing.ViTImageDataset import LabelType
from torch.utils.data import DataLoader
from tboard.summarywriter import write_summary

SEED = int(os.getenv("SEED", "123"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))


class ViTTrainer:
    """
    A trainer class for fine-tuning a Vision Transformer (ViT) model for a task
    involving both bounding box regression and classification using early stopping.

    Assumes the ViT model has separate heads for bounding box prediction
    (bbox_head) and classification (cls_head).
    """

    def __init__(self, model: ViT, device: torch.device, learning_rate: float = 0.001,
                 epochs: int = 5, batch_size: int = BATCH_SIZE, patience: int = 2,
                 annealing_rate: float = 0.000001) -> None:
        """
        Initializes the ViTTrainer.

        Args:
            model (ViT): The Vision Transformer model to train.
            device (torch.device): The device to train on (e.g., 'cuda' or 'cpu').
            learning_rate (float, optional): The initial learning rate for the optimizer. Defaults to 0.001.
            epochs (int, optional): The total number of training epochs (or more accurately,
                                    the number of distinct K-Fold splits to use for training).
                                    Defaults to 5.
            batch_size (int, optional): The batch size for data loaders. Defaults to 32.
            patience (int, optional): The number of epochs with no improvement on validation
                                      loss after which training will be stopped. Defaults to 2.
            annealing_rate (float, optional): The initial learning rate for the cosine anealer. Defaults to 0.000001.
        """
        self.model = model
        self.model.to(device=device)
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self._logger = logging.getLogger(self.__class__.__name__)
        self.patience = patience
        self.annealing_rate = annealing_rate

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> float:
        """
        Executes the main training and validation loop for the model.

        Args:
            train_loader (DataLoader): The data loader for the training set.
            val_loader (DataLoader): The data loader for the validation set.

        Returns:
            float: The best validation loss achieved during training.
        """
        writer = write_summary(run_name="ViT_Trainer")
        optimizer = torch.optim.AdamW(
            [p for p in self.model.cls_head.parameters()] + [p for p in self.model.bbox_head.parameters()],
            lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.annealing_rate)

        bbox_criterion = torchvision.ops.distance_box_iou_loss
        cls_criterion = torch.nn.CrossEntropyLoss()

        best_val_loss = float("inf")

        current_patience = 0
        for epoch in range(self.epochs):
            bbox_loss, cls_los, val_bbox_loss, val_cls_loss = self.train_epoch(
                optimizer, bbox_criterion, cls_criterion, train_loader, val_loader)
            writer.add_scalar("Train/BBox Loss", bbox_loss.item(), epoch)
            writer.add_scalar("Train/Cls Loss", cls_los.item(), epoch)
            scheduler.step()

            val_loss = float((val_bbox_loss + val_cls_loss).item())
            writer.add_scalar("Val/BBox Loss", val_bbox_loss.item(), epoch)
            writer.add_scalar("Val/Cls Loss", val_cls_loss.item(), epoch)
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                current_patience = 0
            else:
                current_patience += 1

            self._logger.info(
                f"Epoch {epoch + 1}/{self.epochs} - " +
                f"Train bbox loss: {bbox_loss.item():.4f} " +
                f"Train cls loss: {cls_los.item():.4f} " +
                f"Val bbox loss: {val_bbox_loss.item():.4f} " +
                f"Val cls loss: {val_cls_loss.item():.4f} " +
                f"Best loss: {best_val_loss:.4f}"
            )

            if current_patience == self.patience:
                self._logger.info(
                    f"Early stopping at epoch {epoch + 1} with patience {self.patience}")
                break

        return best_val_loss

    def train_epoch(self, optimizer: torch.optim.AdamW,  bbox_criterion: torchvision.ops.distance_box_iou_loss,
                    cls_criterion: torch.nn.CrossEntropyLoss,  train_loader: DataLoader,
                    val_loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Trains and validates the model for one K-Fold split (equivalent to one
        epoch in the main training loop).

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            bbox_criterion (Callable): The loss function for bounding box regression.
                                    Expected to take predicted and target boxes.
            cls_criterion (torch.nn.CrossEntropyLoss): The loss function for classification.
                                                    Expected to take predicted logits and target classes.
            train_loader (DataLoader): The data loader for the training subset of the current split.
            val_loader (DataLoader): The data loader for the validation subset of the current split.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            Returns the average training bbox loss, average training classification loss,
            average validation bbox loss, and average validation classification loss
            over the entire dataset subset for the split.
        """
        self.model.train()

        running_train_bbox_loss = 0.0
        running_train_cls_loss = 0.0
        train_batches = 0

        images: torch.Tensor
        labels: LabelType
        bbox: torch.Tensor
        cls: torch.Tensor

        for images, labels in train_loader:
            optimizer.zero_grad()
            input_bbox = labels["bbox"].squeeze().to(self.device)
            input_cls = torch.argmax(labels["cls"], dim=1).to(self.device)

            images = images.to(self.device)
            bbox, cls = self.model(images)

            bbox_loss: torch.Tensor = bbox_criterion(bbox, input_bbox, reduction="mean")
            cls_loss: torch.Tensor = cls_criterion(cls, input_cls)
            loss: torch.Tensor = bbox_loss + cls_loss

            loss.backward()
            optimizer.step()

            running_train_bbox_loss += bbox_loss.item()
            running_train_cls_loss += cls_loss.item()
            train_batches += 1

        avg_train_bbox_loss = torch.tensor(running_train_bbox_loss / train_batches, device=self.device)
        avg_train_cls_loss = torch.tensor(running_train_cls_loss / train_batches, device=self.device)

        self.model.eval()

        running_val_bbox_loss = 0.0
        running_val_cls_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for images, labels in val_loader:
                input_bbox = labels["bbox"].squeeze().to(self.device)
                input_cls = torch.argmax(labels["cls"], dim=1).to(self.device)
                images = images.to(self.device)

                val_bbox_preds, val_cls_preds = self.model(images)

                val_bbox_loss: torch.Tensor = bbox_criterion(val_bbox_preds, input_bbox, reduction="mean")
                val_cls_loss: torch.Tensor = cls_criterion(val_cls_preds, input_cls)

                running_val_bbox_loss += val_bbox_loss.item()
                running_val_cls_loss += val_cls_loss.item()
                val_batches += 1

        avg_val_bbox_loss = torch.tensor(running_val_bbox_loss / val_batches, device=self.device)
        avg_val_cls_loss = torch.tensor(running_val_cls_loss / val_batches, device=self.device)

        return avg_train_bbox_loss, avg_train_cls_loss, avg_val_bbox_loss, avg_val_cls_loss
