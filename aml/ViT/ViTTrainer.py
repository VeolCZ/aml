import logging
import math
import multiprocessing
import os
from typing import Any, Generator
import torch
import copy
import torchvision
from ViT.ViT import ViT
from sklearn.model_selection import RepeatedKFold
from torch.utils.data import DataLoader, Subset
from preprocessing.ViTImageDataset import LabelType, ViTImageDataset


class ViTTrainer:
    def __init__(self, model: ViT, device: torch.device, dataset: ViTImageDataset, lr: float = 0.001, n_splits: int = 5, epochs: int = 5, batch_size: int = 32) -> None:
        self.model = model
        self.model.to(device=device)
        self.device = device
        self.dataset = dataset
        self.lr = lr
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.epochs = epochs
        self._logger = logging.getLogger(self.__class__.__name__)
        self.SEED = int(os.getenv("SEED", 123))
        torch.manual_seed(self.SEED)

    def get_loaders(self) -> Generator[tuple[DataLoader, DataLoader], None, None]:
        kfold = RepeatedKFold(n_splits=self.n_splits, n_repeats=math.ceil(self.epochs/self.n_splits),
                              random_state=self.SEED)

        for train_idx, val_idx in kfold.split(self.dataset):
            train_subset = Subset(self.dataset, train_idx)
            val_subset = Subset(self.dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True,
                                      num_workers=multiprocessing.cpu_count(), pin_memory=self.device.type == "cuda")
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=True,
                                    num_workers=multiprocessing.cpu_count(), pin_memory=self.device.type == "cuda")

            yield train_loader, val_loader

    def train(self, model_path: str, save: bool = False) -> None:
        optimizer = torch.optim.AdamW(
            [p for p in self.model.cls_head.parameters()] + [p for p in self.model.bbox_head.parameters()], lr=self.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr / 100)

        bbox_criterion = torchvision.ops.complete_box_iou_loss
        cls_criterion = torch.nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_model = copy.deepcopy(self.model.state_dict())

        loader_generator = self.get_loaders()
        for epoch in range(self.epochs):
            train_loader, val_loader = next(loader_generator)

            bbox_loss, cls_los, val_bbox_loss, val_cls_loss = self.train_epoch(
                optimizer, scheduler, bbox_criterion, cls_criterion, train_loader, val_loader)

            val_loss = float((val_bbox_loss + val_cls_loss).item())
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.model.state_dict())

            self._logger.info(
                f"Epoch {epoch + 1}/{self.epochs} - "
                f"Train bbox loss: {bbox_loss.item():.4f}, "
                f"Train cls loss: {cls_los.item():.4f}, "
                f"Val bbox loss: {val_bbox_loss.item():.4f}, "
                f"Val cls loss: {val_cls_loss.item():.4f}"
            )

        if best_model:
            self.model.load_state_dict(best_model)
        if save:
            torch.save(best_model, model_path +
                       f"ValLoss_{round(val_loss, 2)}" + ".pth")

    def train_epoch(self, optimizer: torch.optim.AdamW, scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,  bbox_criterion: torchvision.ops.complete_box_iou_loss, cls_criterion: torch.nn.CrossEntropyLoss,  train_loader: DataLoader, val_loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Train the model for one epoch.

        Args:
            epoch (int): Current epoch number.
        """
        self.model.train()
        images: torch.Tensor
        labels: LabelType
        for images, labels in train_loader:
            optimizer.zero_grad()
            input_bbox = labels["bbox"].squeeze().to(self.device)
            input_cls = torch.argmax(labels["cls"], dim=1).to(self.device)
            images = images.to(self.device)
            bbox, cls = self.model(images)
            bbox_loss: torch.Tensor = bbox_criterion(bbox, input_bbox).mean()
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
                val_bbox_loss: torch.Tensor = bbox_criterion(
                    val_bbox, input_bbox).mean()
                val_cls_loss: torch.Tensor = cls_criterion(
                    val_cls, input_cls)

        scheduler.step()
        return bbox_loss, cls_loss, val_bbox_loss, val_cls_loss
