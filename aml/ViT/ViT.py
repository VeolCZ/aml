from transformers import ViTForImageClassification
import torch


class ViT(torch.nn.Module):
    def __init__(self, hidden_size: int = 100) -> None:
        super(ViT, self).__init__()
        dp_rate = 0.2

        self.backbone = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224", cache_dir="/data/vit")
        for param in self.backbone.parameters():
            param.requires_grad = False
        # self.backbone.classifier = torch.nn.Identity()
        backbone_out_size = 1000  # 768
        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(backbone_out_size, hidden_size),
            torch.nn.Dropout(dp_rate),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, 200),
            # TODO abstact the number of classes
            torch.nn.Softmax(dim=1),
        )
        self.bbox_head = torch.nn.Sequential(
            torch.nn.Linear(backbone_out_size, hidden_size),
            torch.nn.Dropout(dp_rate),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, 4),
        )

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        backbone = self.backbone(pixel_values=x).logits
        bbox = self.bbox_head(backbone)
        cls = self.cls_head(backbone)
        return bbox, cls
