import torch
import torchvision
from sklearn.metrics import accuracy_score
from interface.ModelInterface import ModelInterface
from torcheval.metrics.functional import multiclass_auroc, multiclass_f1_score, multiclass_confusion_matrix
from typing import Dict


class Evaluator:
    @staticmethod
    def get_accuracy(CLS: torch.Tensor, true_label_indices: torch.Tensor) -> float:
        """
        Calculate the accuracy of the model predictions.
        :param CLS: The model predictions.
        :param true_label_indices: The true labels.
        :return: The accuracy score.
        """
        return float(accuracy_score(true_label_indices, CLS.argmax(dim=1)))

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
    def get_top_k(CLS: torch.Tensor, true_label_indices, k: int) -> float:
        """
        Calculate the top-k accuracy of the model predictions.
        :param CLS: The model predictions.
        :param true_label_indices: The true labels.
        :param k: The number of top predictions to consider.
        :return: The top-k accuracy score.
        """
        true_label_indices = true_label_indices.unsqueeze(dim=1)
        _, top_k_indices = torch.topk(CLS, k=k, dim=1)

        comparison_result = torch.eq(true_label_indices, top_k_indices)
        is_in_top_k = torch.any(comparison_result, dim=1)

        return float(is_in_top_k.float().mean())

    @staticmethod
    def multiroc(CLS: torch.Tensor, true_label_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the multi-class ROC AUC score.
        :param CLS: The model predictions.
        :param true_label_indices: The true labels.
        :return: The ROC AUC score.
        """
        CLS = CLS.squeeze(-1)
        return float(multiclass_auroc(CLS, true_label_indices, num_classes=CLS.shape[1]))

    @staticmethod
    def f1_score(CLS: torch.Tensor, true_label_indices: torch.Tensor):
        """
        Calculate the F1 score for the model predictions.
        :param CLS: The model predictions.
        :param true_label_indices: The true labels.
        :return: The F1 score.
        """
        CLS = CLS.squeeze(-1)
        return float(multiclass_f1_score(CLS, true_label_indices, num_classes=CLS.shape[1], average="weighted"))

    @staticmethod
    def confusion_matrix(CLS: torch.Tensor, true_label_indices: torch.Tensor) -> torch.Tensor:
        """
        Calculate the confusion matrix for the model predictions.
        :param cls: The model predictions.
        :param true_label_indices: The true labels.
        :return: The confusion matrix.
        """
        CLS = CLS.squeeze(-1)
        return multiclass_confusion_matrix(CLS, true_label_indices, num_classes=CLS.shape[1], normalize="all")

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
            "confusion_matrix": confusion_matrix,
            "num_classes": cls.shape[1]
        }
        return eval_results
