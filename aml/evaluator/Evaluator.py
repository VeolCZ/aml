import torch
import torchvision
from sklearn.metrics import accuracy_score
from interface.ModelInterface import ModelInterface
from torcheval.metrics.functional import multiclass_auroc, multiclass_f1_score, multiclass_confusion_matrix


# PLEASE REFRACTOR ME SO THE INPUTS TO EVERY FUNCTION ARE: BBOX, CLS, other args.. so that we save a bunch of compute by not recalculating the predictions for every metric

class Evaluator:
    @staticmethod
    def get_accuracy(cls, true_label_indices: torch.Tensor) -> float:
        #_, cls = model.predict(input_data)
        #print(cls.shape)
        return float(accuracy_score(true_label_indices, cls.argmax(dim=1)))

    @staticmethod
    def get_IOU(model: ModelInterface, input_data: torch.Tensor, true_label: torch.Tensor) -> float:
        bbox, _ = model.predict(input_data)
        return float(torchvision.ops.complete_box_iou_loss(bbox, true_label, reduction="mean").item())

    @staticmethod
    def get_top_k(cls:torch.Tensor, true_label_indices, k: int) -> float:
        #_, cls = model.predict(input_data)
        #true_label_indices = torch.argmax(true_label, dim=1).unsqueeze(dim=1)
        true_label_indices = true_label_indices.unsqueeze(dim=1)
        _, top_k_indices = torch.topk(cls, k=k, dim=1)

        comparison_result = torch.eq(true_label_indices, top_k_indices)
        is_in_top_k = torch.any(comparison_result, dim=1)

        return float(is_in_top_k.float().mean())
    
    @staticmethod
    def multiroc(cls:torch.Tensor, true_label_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        #_, cls = model.predict(input_data)
        #true_label_indices = torch.argmax(true_label, dim=1)
        cls = cls.squeeze(-1)
        return float(multiclass_auroc(cls, true_label_indices, num_classes=cls.shape[1]))
        
    @staticmethod
    def f1_score(cls:torch.Tensor, true_label_indices: torch.Tensor):
        #_, cls = model.predict(input_data)
        #true_label_indices = torch.argmax(true_label, dim=1)
        cls = cls.squeeze(-1)
        return float(multiclass_f1_score(cls, true_label_indices, num_classes=cls.shape[1], average="weighted"))

    @staticmethod
    def confusion_matrix(cls:torch.Tensor, true_label_indices: torch.Tensor) -> torch.Tensor:
        #_, cls = model.predict(input_data)
        #true_label_indices = torch.argmax(true_label, dim=1)
        cls = cls.squeeze(-1)
        return multiclass_confusion_matrix(cls, true_label_indices, num_classes=cls.shape[1], normalize="all")
    
    @staticmethod
    def classifier_eval(model: ModelInterface, input_data:torch.Tensor, true_label: torch.Tensor):
        _, cls = model.predict(input_data)
        true_label_indices = torch.argmax(true_label, dim=1)
        acc = Evaluator.get_accuracy(cls, true_label_indices)
        top_k = Evaluator.get_top_k(cls, true_label_indices, k=5)
        multiroc = Evaluator.multiroc(cls, true_label_indices)
        f1 = Evaluator.f1_score(cls, true_label_indices)
        confusion_matrix = Evaluator.confusion_matrix(cls, true_label_indices)
        eval_results = {
            "accuracy": acc,
            "top_k": top_k,
            "multiroc": multiroc,
            "f1_score": f1,
            "confusion_matrix": confusion_matrix
        }
        return eval_results
        # nor needed, do it in the function special_needs_indices = torch.argmax(true_label, dim=1).unsqueeze(dim=1) #because top_k
        
