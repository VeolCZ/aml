from typing import Dict
import torch
import torchvision
from sklearn.metrics import accuracy_score
from interface.ModelInterface import ModelInterface
from torcheval.metrics.functional import multiclass_auroc, multiclass_f1_score, multiclass_confusion_matrix, multiclass_accuracy


class Evaluator:
    @staticmethod
    def get_accuracy(cls: torch.Tensor, true_label_indices: torch.Tensor) -> float:
        """
        Calculate the accuracy of the model predictions.
        :param CLS: The model predictions.
        :param true_label_indices: The true labels.
        :return: The accuracy score.
        """
        return float(accuracy_score(true_label_indices, cls.argmax(dim=1)))

    @staticmethod
    def get_IOU(model: ModelInterface, input_data: torch.Tensor, true_label: torch.Tensor) -> float:
        """
        Calculate the Intersection over Union (IoU) for the model predictions.
        :param model: The model to evaluate.
        :param input_data: The input data for the model.
        :param true_label: The true labels.
        :return: The IoU score.
        """
        bbox, _ = model.predict(input_data)
        return float(torchvision.ops.complete_box_iou_loss(bbox, true_label, reduction="mean").item())

    @staticmethod
    def get_top_k(cls: torch.Tensor, true_label_indices: torch.Tensor, k: int) -> float:
        """
        Calculate the top-k accuracy of the model predictions.
        :param CLS: The model predictions.
        :param true_label_indices: The true labels.
        :param k: The number of top predictions to consider.
        :return: The top-k accuracy score.
        """
        true_label_indices = true_label_indices.unsqueeze(dim=1)
        _, top_k_indices = torch.topk(cls, k=k, dim=1)

        comparison_result = torch.eq(true_label_indices, top_k_indices)
        is_in_top_k = torch.any(comparison_result, dim=1)

        return float(is_in_top_k.float().mean())

    @staticmethod
    def multiroc(cls: torch.Tensor, true_label_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the multi-class ROC AUC score.
        :param CLS: The model predictions.
        :param true_label_indices: The true labels.
        :return: The ROC AUC score.
        """
        return float(multiclass_auroc(cls, true_label_indices, num_classes=cls.shape[1]))

    @staticmethod
    def f1_score(cls: torch.Tensor, true_label_indices: torch.Tensor):
        """
        Calculate the F1 score for the model predictions.
        :param CLS: The model predictions.
        :param true_label_indices: The true labels.
        :return: The F1 score.
        """
        return float(multiclass_f1_score(cls, true_label_indices, num_classes=cls.shape[1], average="weighted"))

    @staticmethod
    def confusion_matrix(cls: torch.Tensor, true_label_indices: torch.Tensor) -> torch.Tensor:
        """
        Calculate the confusion matrix for the model predictions.
        :param cls: The model predictions.
        :param true_label_indices: The true labels.
        :return: The confusion matrix.
        """
        return multiclass_confusion_matrix(cls, true_label_indices, num_classes=cls.shape[1], normalize="all")

    @staticmethod
    def best_and_worst(pred_label_indices: torch.Tensor, true_label_indices: torch.Tensor, num_classes: int, k: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computing top k best and worst performing glasses based on the per-class accuracy.
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
    def classifier_eval(model: ModelInterface, input_data: torch.Tensor, true_label: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate the model using various metrics.
        :param model: The model to evaluate.
        :param input_data: The input data for the model.
        :param true_label: The true labels.
        :return: A dictionary containing the evaluation results.
        """
        _, cls = model.predict(input_data)
        true_label_indices = torch.argmax(true_label, dim=-1)

        acc = Evaluator.get_accuracy(cls, true_label_indices)
        top_k = Evaluator.get_top_k(cls, true_label_indices, k=5)
        multiroc = Evaluator.multiroc(cls, true_label_indices)
        f1 = Evaluator.f1_score(cls, true_label_indices)
        confusion_matrix = Evaluator.confusion_matrix(cls, true_label_indices)
        best, worst = Evaluator.best_and_worst(cls, true_label_indices, cls.shape[1], k=3)

        eval_results = {
            "accuracy": acc,
            "top_k": top_k,
            "multiroc": multiroc,
            "f1_score": f1,
            "confusion_matrix": confusion_matrix,
            "num_classes": cls.shape[1],
            "best_classes": best,
            "worst_classes": worst,
        }
        return eval_results
