from dataclasses import dataclass
import torch
import torchvision
from sklearn.metrics import accuracy_score
from interface.ModelInterface import ModelInterface
from torcheval.metrics.functional import multiclass_auroc, multiclass_f1_score, multiclass_confusion_matrix, \
    multiclass_accuracy


@dataclass
class EvalMetric:
    accuracy: float
    top_3: float
    top_5: float
    multiroc: float
    f1_score: float
    confusion_matrix: torch.Tensor
    num_classes: int
    best_classes: torch.Tensor
    worst_classes: torch.Tensor
    iou: float


class Evaluator:
    @staticmethod
    def get_accuracy(clas: torch.Tensor, true_label_indices: torch.Tensor) -> float:
        """
        Calculate the accuracy of the model predictions.
        args:
            clas (torch.Tensor): The model predictions.
            true_label_indices (torch.Tensor): The true labels.
        returns: The accuracy score.
        """
        return float(accuracy_score(true_label_indices, clas.argmax(dim=1)))

    @staticmethod
    def get_IOU(bbox: torch.Tensor, true_label: torch.Tensor) -> float:
        """
        Calculate the Intersection over Union (IoU) for the model predictions.
        args:
            bbox (torch.Tensor): The predicted bounding boxes.
            true_label (torch.Tensor): The true bounding boxes.
        returns: The IoU score.
        """
        return float(torchvision.ops.complete_box_iou_loss(bbox, true_label, reduction="mean").item())

    @staticmethod
    def get_top_k(clas: torch.Tensor, true_label_indices: torch.Tensor, k: int) -> float:
        """
        Calculate the top-k accuracy of the model predictions.
        args:
            clas (torch.Tensor): The model predictions.
            true_label_indices (torch.Tensor): The true labels.
            k (int): The number of top predictions to consider.
        returns: The top-k accuracy score.
        """
        true_label_indices = true_label_indices.unsqueeze(dim=1)
        _, top_k_indices = torch.topk(clas, k=k, dim=1)

        comparison_result = torch.eq(true_label_indices, top_k_indices)
        is_in_top_k = torch.any(comparison_result, dim=1)

        return float(is_in_top_k.float().mean())

    @staticmethod
    def multiroc(clas: torch.Tensor, true_label_indices: torch.Tensor) -> float:
        """
        Calculate the multi-class ROC AUC score.
        args:
            clas (torch.Tensor): The model predictions.
            true_label_indices (torch.Tensor): The true labels.
        returns: The multi-class ROC AUC score.
        """
        return float(multiclass_auroc(clas, true_label_indices, num_classes=clas.shape[1]))

    @staticmethod
    def f1_score(clas: torch.Tensor, true_label_indices: torch.Tensor) -> float:
        """
        Calculate the F1 score for the model predictions.
        args:
            clas (torch.Tensor): The model predictions.
            true_label_indices (torch.Tensor): The true labels.
        returns: The F1 score.
        """
        return float(multiclass_f1_score(clas, true_label_indices, num_classes=clas.shape[1], average="weighted"))

    @staticmethod
    def confusion_matrix(clas: torch.Tensor, true_label_indices: torch.Tensor) -> torch.Tensor:
        """
        Calculate the confusion matrix for the model predictions.
        args:
            clas (torch.Tensor): The model predictions.
            true_label_indices (torch.Tensor): The true labels.
        returns: The confusion matrix.
        """
        return multiclass_confusion_matrix(clas, true_label_indices, num_classes=clas.shape[1], normalize="pred")

    @staticmethod
    def best_and_worst(pred_label_indices: torch.Tensor,
                       true_label_indices: torch.Tensor,
                       num_classes: int, k: int = 3
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computing top k best and worst performing glasses based on the per-class accuracy.
        args:
            pred_label_indices (torch.Tensor): The predicted labels.
            true_label_indices (torch.Tensor): The true labels.
            num_classes (int): The number of classes.
            k (int): The number of best and worst classes to return.
        returns: A tuple containing the indices of the best and worst classes.
        """
        per_class_acc = multiclass_accuracy(
            input=pred_label_indices, target=true_label_indices, num_classes=num_classes, average=None)
        valid = ~torch.isnan(per_class_acc)
        valid_accs = per_class_acc[valid]
        k = min(k, len(valid_accs))
        best = torch.argsort(valid_accs, descending=True)[:k]
        worst = torch.argsort(valid_accs, descending=False)[:k]
        return best, worst

    @staticmethod
    def eval(model: ModelInterface, input_data: torch.Tensor,
             clas_label: torch.Tensor, bbox_label: torch.Tensor
             ) -> EvalMetric:
        """
        Evaluate the model using various metrics.
        args:
            model (ModelInterface): The model to evaluate.
            input_data (torch.Tensor): The input data for the model.
            clas_label (torch.Tensor): The true class labels.
            bbox_label (torch.Tensor): The true bounding box labels.
        returns: An EvalMetric object containing the evaluation results.
        """
        bbox, clas = model.predict(input_data)
        true_label_indices = torch.argmax(clas_label, dim=-1)

        acc = Evaluator.get_accuracy(clas, true_label_indices)
        top_3 = Evaluator.get_top_k(clas, true_label_indices, k=3)
        top_5 = Evaluator.get_top_k(clas, true_label_indices, k=5)
        multiroc = Evaluator.multiroc(clas, true_label_indices)
        f1 = Evaluator.f1_score(clas, true_label_indices)
        confusion_matrix = Evaluator.confusion_matrix(clas, true_label_indices)
        best, worst = Evaluator.best_and_worst(clas, true_label_indices, clas.shape[1], k=3)

        iou = Evaluator.get_IOU(bbox, bbox_label)
        eval_results = EvalMetric(
            accuracy=acc,
            top_3=top_3,
            top_5=top_5,
            multiroc=multiroc,
            f1_score=f1,
            confusion_matrix=confusion_matrix,
            num_classes=clas.shape[0],
            best_classes=best,
            worst_classes=worst,
            iou=iou
        )
        return eval_results
