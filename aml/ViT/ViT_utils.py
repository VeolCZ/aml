import os
import torch
import multiprocessing
from datetime import datetime
from ViT.ViT import ViT
from ViT.ViTTrainer import ViTTrainer
from torch.utils.data import DataLoader
from torch.utils.data import random_split, Subset
from preprocessing.ViTImageDataset import ViTImageDataset


def train_vit() -> None:
    # Config
    SEED = int(os.getenv("SEED", 123))
    torch.manual_seed(SEED)
    batch_size = 400
    device = torch.device("cuda")
    epochs = 1

    # Datasets
    dataset = ViTImageDataset(type="train")

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_indices, val_indices, test_indices = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_dataset = Subset(dataset, train_indices.indices)
    val_dataset = Subset(dataset, val_indices.indices)
    test_dataset = Subset(dataset, test_indices.indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=multiprocessing.cpu_count(), pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=multiprocessing.cpu_count(), pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=multiprocessing.cpu_count(), pin_memory=device.type == "cuda")

    # Train the model
    model = ViT()
    trainer = ViTTrainer(model, train_loader, val_loader, device)
    trainer.train(
        epochs=epochs, model_path=f"/data/ViT_{datetime.utcnow()}")
