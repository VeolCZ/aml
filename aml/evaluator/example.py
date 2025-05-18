from sklearn.base import accuracy_score
import torch
from ViT.ViT import ViT
from torch.utils.data import DataLoader
from interface.ModelInterface import ModelInterface
from preprocessing.ViTImageDataset import ViTImageDataset

# will work for anything that implements the ModelInterface


def get_accuracy(model: ModelInterface, input_data: torch.Tensor, true_data: torch.Tensor) -> float:
    _, cls = model.predict(input_data)
    return float(accuracy_score(true_data, cls))


if __name__ == "__main__":
    model = ViT()
    # load weights
    # model.load_state_dict...

    eval_dataset = ViTImageDataset(type="eval")
    test_loader = DataLoader(eval_dataset, batch_size=2)
    for img, cls in test_loader:
        acc = get_accuracy(model, img, cls)
