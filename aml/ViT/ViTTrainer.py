import torch
import copy
import torchvision
from ViT.ViT import ViT
from preprocessing.ViTImageDataset import LabelType
from torch.utils.data import DataLoader


class ViTTrainer:
    def __init__(self, model: ViT, train_loader: DataLoader, val_loader: DataLoader, device: str = "cuda"):
        self.model = model
        self.model.to(device=device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def train(self, epochs: int, model_path: str) -> None:
        optimizer = torch.optim.AdamW(
            [p for p in self.model.cls_head.parameters()] + [p for p in self.model.bbox_head.parameters()], lr=0.01)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=0.001)

        bbox_criterion = torchvision.ops.complete_box_iou_loss
        cls_criterion = torch.nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        best_model = copy.deepcopy(self.model.state_dict())

        images: torch.Tensor
        labels: LabelType

        for epoch in range(epochs):
            print(f"Training epoch {epoch}")
            self.model.train()
            for images, labels in self.train_loader:
                optimizer.zero_grad()

                input_bbox = labels["bbox"].squeeze().to(self.device)
                input_cls = labels["cls"].squeeze().to(self.device)
                images = images.to(self.device)

                bbox, cls = self.model(images)
                bbox_loss = bbox_criterion(bbox, input_bbox).mean()
                cls_loss = cls_criterion(cls, input_cls)
                loss = bbox_loss + cls_loss

                loss.backward()
                optimizer.step()
                scheduler.step()

            print(f"Evaluating epoch {epoch}")
            self.model.eval()
            for images, labels in self.val_loader:
                with torch.no_grad():
                    input_bbox = labels["bbox"].squeeze().to(self.device)
                    input_cls = labels["cls"].squeeze().to(self.device)
                    images = images.to(self.device)

                    val_bbox, val_cls = self.model(images)
                    print("val_cls.shape", val_cls.shape)
                    print("input_cls.shape", input_cls.shape)

                    val_bbox_loss = bbox_criterion(val_bbox, input_bbox).mean()
                    val_cls_loss = cls_criterion(val_cls, input_cls)
                    val_loss = val_bbox_loss + val_cls_loss

            print(
                f"Epoch [{epoch}] Train Loss - Bbox:{bbox_loss.item()} Cls:{cls_loss.item()}, Val Loss - Bbox:{val_bbox_loss.item()} Cls:{val_cls_loss.item()}")

            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.model.state_dict())

        if best_model:
            self.model.load_state_dict(best_model)
            torch.save(best_model, model_path)
