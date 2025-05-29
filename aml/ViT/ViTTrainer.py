import logging
import math
import os
import torch
import copy
import torchvision
import multiprocessing
from ViT.ViT import ViT
from typing import Generator
from sklearn.model_selection import RepeatedKFold
from torch.utils.data import DataLoader, Subset
from preprocessing.ViTImageDataset import LabelType, ViTImageDataset

SEED = int(os.getenv("SEED", "123"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))


class ViTTrainer:
    """
    A trainer class for fine-tuning a Vision Transformer (ViT) model for a task
    involving both bounding box regression and classification using K-Fold
    cross-validation and early stopping.

    Assumes the ViT model has separate heads for bounding box prediction
    (bbox_head) and classification (cls_head).
    """

    def __init__(self, model: ViT, device: torch.device, dataset: ViTImageDataset, learning_rate: float = 0.001,
                 n_splits: int = 5, epochs: int = 5, batch_size: int = BATCH_SIZE, patience: int = 2,
                 annealing_rate: float = 0.000001) -> None:
        """
        Initializes the ViTTrainer.

        Args:
            model (ViT): The Vision Transformer model to train.
            device (torch.device): The device to train on (e.g., 'cuda' or 'cpu').
            dataset (ViTImageDataset): The dataset to use for training and validation.
            learning_rate (float, optional): The initial learning rate for the optimizer. Defaults to 0.001.
            n_splits (int, optional): The number of splits for K-Fold cross-validation. Defaults to 5.
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
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.epochs = epochs
        self._logger = logging.getLogger(self.__class__.__name__)
        self.patience = patience
        self.annealing_rate = annealing_rate

    def get_loaders(self) -> Generator[tuple[DataLoader, DataLoader], None, None]:
        """
        Generates pairs of training and validation DataLoaders for each split
        of Repeated K-Fold cross-validation.

        The number of repeats is calculated such that there are at least
        'epochs' total splits generated. The main training loop in train
        will consume exactly epochs splits from this generator.

        Yields:
            Generator[Tuple[DataLoader, DataLoader], None, None]: A generator
            yielding tuples of (train_loader, val_loader) for each split.

        NOTE: For optimal performance set epochs to be fully divisable by n_splits
        """
        kfold = RepeatedKFold(n_splits=self.n_splits, n_repeats=math.ceil(self.epochs/self.n_splits),
                              random_state=SEED)

        for train_idx, val_idx in kfold.split(self.dataset):
            train_subset = Subset(self.dataset, train_idx)
            val_subset = Subset(self.dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True,
                                      num_workers=multiprocessing.cpu_count(), pin_memory=self.device.type == "cuda")
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=True,
                                    num_workers=multiprocessing.cpu_count(), pin_memory=self.device.type == "cuda")

            yield train_loader, val_loader

    def train(self, model_path: str = "/weights/model", save: bool = False) -> float:
        """
        Trains the ViT model using the configured K-Fold cross-validation setup
        and early stopping.

        Optimizes only the parameters of the classification and bounding box
        heads of the model.

        Args:
            model_path (str, optional): The base path to save the best model checkpoint.
                                        Defaults to "/weights/model".
            save (bool, optional): Whether to save the best model checkpoint. Defaults to False.

        Returns:
            float: The best validation loss achieved during training.

        Raises:
            StopIteration: If the loader generator runs out of splits before
                           completing the specified number of 'epochs' (splits).
        """
        optimizer = torch.optim.AdamW(
            [p for p in self.model.cls_head.parameters()] + [p for p in self.model.bbox_head.parameters()],
            lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.annealing_rate)

        bbox_criterion = torchvision.ops.complete_box_iou_loss
        cls_criterion = torch.nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_model = copy.deepcopy(self.model.state_dict())

        current_patience = 0
        loader_generator = self.get_loaders()
        for epoch in range(self.epochs):
            train_loader, val_loader = next(loader_generator)

            bbox_loss, cls_los, val_bbox_loss, val_cls_loss = self.train_epoch(
                optimizer, bbox_criterion, cls_criterion, train_loader, val_loader)
            scheduler.step()

            val_loss = float((val_bbox_loss + val_cls_loss).item())
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.model.state_dict())
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

        if best_model:
            self.model.load_state_dict(best_model)
        if save:
            torch.save(best_model, model_path +
                       f"ValLoss_{round(best_val_loss, 3)}" + ".pth")

        return best_val_loss

    def train_epoch(self, optimizer: torch.optim.AdamW,  bbox_criterion: torchvision.ops.complete_box_iou_loss,
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
            Returns the training bbox loss, training classification loss,
            validation bbox loss, and validation classification loss.

            NOTE: These are the losses from the last batch of the
            respective loops, not the average over the entire dataset subset.
        """
        self.model.train()
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

        self.model.eval()
        val_bbox: torch.Tensor
        val_cls: torch.Tensor
        for images, labels in val_loader:
            with torch.no_grad():
                input_bbox = labels["bbox"].squeeze().to(self.device)
                input_cls = torch.argmax(labels["cls"], dim=1).to(self.device)
                images = images.to(self.device)
                val_bbox, val_cls = self.model(images)
                val_bbox_loss: torch.Tensor = bbox_criterion(val_bbox, input_bbox, reduction="mean")
                val_cls_loss: torch.Tensor = cls_criterion(
                    val_cls, input_cls)

        return bbox_loss, cls_loss, val_bbox_loss, val_cls_loss
