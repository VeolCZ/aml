import logging
import torch
import copy
import torchvision
from ViT.ViT import ViT
from preprocessing.ViTImageDataset import LabelType
from torch.utils.data import DataLoader


class ViTTrainer:
    def __init__(self, model: ViT, train_loader: DataLoader, val_loader: DataLoader, device: torch.device):
        self.model = model
        self.model.to(device=device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self._logger = logging.getLogger(self.__class__.__name__)

    def train(self, epochs: int, model_path: str, save: bool = False) -> None:
        optimizer = torch.optim.AdamW(
            [p for p in self.model.cls_head.parameters()] + [p for p in self.model.bbox_head.parameters()], lr=0.01)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=0.0005)

        bbox_criterion = torchvision.ops.complete_box_iou_loss
        cls_criterion = torch.nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_model = copy.deepcopy(self.model.state_dict())

        images: torch.Tensor
        labels: LabelType

        for epoch in range(epochs):
            self.model.train()

            for images, labels in self.train_loader:
                optimizer.zero_grad()

                input_bbox = labels["bbox"].squeeze().to(self.device)
                input_cls = torch.argmax(labels["cls"], dim=1).to(self.device)
                images = images.to(self.device)

                bbox, cls = self.model(images)
                bbox_loss = bbox_criterion(bbox, input_bbox).mean()
                cls_loss: torch.Tensor = cls_criterion(cls, input_cls)
                loss = bbox_loss + cls_loss

                loss.backward()
                optimizer.step()

            self._logger.info(f"Epoch [{epoch}] Train Loss - Bbox:{bbox_loss.item()} Cls:{cls_loss.item()}")

            self.model.eval()
            val_bbox: torch.Tensor
            val_cls: torch.Tensor
            for images, labels in self.val_loader:
                with torch.no_grad():
                    input_bbox = labels["bbox"].squeeze().to(self.device)
                    input_cls = torch.argmax(labels["cls"], dim=1).to(self.device)
                    images = images.to(self.device)

                    val_bbox, val_cls = self.model(images)
                    val_bbox_loss = bbox_criterion(val_bbox, input_bbox).mean()
                    val_cls_loss: torch.Tensor = cls_criterion(val_cls, input_cls)
                    val_loss = val_bbox_loss + val_cls_loss

            self._logger.info(f"Epoch [{epoch}] Val Loss - Bbox:{val_bbox_loss.item()} Cls:{val_cls_loss.item()}")

            scheduler.step()
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.model.state_dict())

        if best_model:
            self.model.load_state_dict(best_model)
        if save:
            torch.save(best_model, model_path+f"ValLoss_{round(val_loss.item(), 2)}" + ".pth")
