import os
import torch
import numpy as np
import multiprocessing
from ViT.ViT import ViT
from datetime import datetime
from ViT.ViTTrainer import ViTTrainer
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from preprocessing.ViTImageDataset import ViTImageDataset


def train_vit() -> None:
    # Config
    SEED = int(os.getenv("SEED", 123))
    torch.manual_seed(SEED)
    batch_size = 350
    device = torch.device("cuda")
    model_path = f"/data/ViT_{datetime.utcnow()}"

    # Datasets
    train_dataset = ViTImageDataset(type="train")
    eval_dataset = ViTImageDataset(type="eval")

    all_labels = train_dataset.get_cls_labels()

    train_indices, test_indices, _, _ = train_test_split(
        np.arange(len(train_dataset)),
        all_labels,
        test_size=0.1,
        stratify=all_labels,
        random_state=SEED
    )

    train_dataset_subset = Subset(train_dataset, train_indices)
    # test_dataset = Subset(eval_dataset, test_indices)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
    #  num_workers=multiprocessing.cpu_count(), pin_memory=device.type == "cuda")

    # Train the model
    model = ViT()
    trainer = ViTTrainer(model, device, train_dataset_subset,
                         epochs=5, n_splits=2, batch_size=150)
    trainer.train(model_path=model_path, save=False)

    # Test the model
    # model.load_state_dict(torch.load("/data/ViT_2025-05-14 16:05:31.936127ValLoss_1.92.pth")).to(device)

    # total = 0
    # correct = 0
    # for images, labels in test_loader:
    # model.eval()
    # images = images.to(device)
    # bbox, cls = model(images)

    # predicted_class_indices = cls.argmax(-1)
    # actual_class_ids_batch = labels["cls"].argmax(-1).to(device)

    # for k_in_batch in range(len(predicted_class_indices)):
    # # assert k_in_batch < 128
    # pred_idx = predicted_class_indices[k_in_batch].item()
    # actual_cls_id = actual_class_ids_batch[k_in_batch].item()

    # if pred_idx == actual_cls_id:
    # correct += 1
    # total += 1
    # print(f"Predicted: {pred_idx}, Actual: {actual_cls_id}")
    # print(f"Accuracy {correct / total}")
