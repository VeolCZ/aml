from datetime import datetime
import multiprocessing
import os
import torch
from torch.utils.data import random_split, Subset
from ViT.ViT import ViT
from ViT.ViTTrainer import ViTTrainer
from preprocessing.ViTImageDataset import ViTImageDataset
from torch.utils.data import DataLoader


def train_vit() -> None:
    # Config
    SEED = int(os.getenv("SEED", 123))
    torch.manual_seed(SEED)
    batch_size = 2
    device = torch.device("cuda")
    epochs = 2

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

    train_dataset = Subset(dataset, train_indices.indices)  # [:1]
    val_dataset = Subset(dataset, val_indices.indices)
    test_dataset = Subset(dataset, test_indices.indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=multiprocessing.cpu_count(), pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True,  # TODO remove drop_last
                            num_workers=multiprocessing.cpu_count(), pin_memory=True if device.type == 'cuda' else False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=multiprocessing.cpu_count(), pin_memory=True if device.type == 'cuda' else False)

    print(f"Train size: {len(train_loader.dataset)}")
    # Train the model
    model = ViT()
    trainer = ViTTrainer(model, train_loader, val_loader, device)
    trainer.train(
        epochs=epochs, model_path=f"/data/ViT_{datetime.utcnow()}.pth")
