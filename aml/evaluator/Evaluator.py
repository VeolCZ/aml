import torch
import torchvision
from sklearn.metrics import accuracy_score
from interface.ModelInterface import ModelInterface


# PLEASE REFRACTOR ME SO THE INPUTS TO EVERY FUNCTION ARE: BBOX, CLS, other args.. so that we save a bunch of compute by not recalculating the predictions for every metric

class Evaluator:
    @staticmethod
    def get_accuracy(model: ModelInterface, input_data: torch.Tensor, true_label: torch.Tensor) -> float:
        _, cls = model.predict(input_data)
        print(f"true_label_shape: {true_label.shape}")
        print(f"cls_shape: {cls.shape}")
        return float(accuracy_score(true_label.argmax(dim=1), cls.argmax(dim=1)))

    @staticmethod
    def get_IOU(model: ModelInterface, input_data: torch.Tensor, true_label: torch.Tensor) -> float:
        bbox, _ = model.predict(input_data)
        return float(torchvision.ops.complete_box_iou_loss(bbox, true_label, reduction="mean").item())

    @staticmethod
    def get_top_k(model: ModelInterface, input_data: torch.Tensor, true_label: torch.Tensor, k: int) -> float:
        _, cls = model.predict(input_data)
        true_label_indices = torch.argmax(true_label, dim=1).unsqueeze(dim=1)
        _, top_k_indices = torch.topk(cls, k=k, dim=1)

        comparison_result = torch.eq(true_label_indices, top_k_indices)
        is_in_top_k = torch.any(comparison_result, dim=1)

        return float(is_in_top_k.float().mean())
