import multiprocessing
import os
import torch
from transformers import ViTForImageClassification, BatchFeature
from preprocessing.ViTImageDataset import LabelType, ViTImageDataset
from torch.utils.data import DataLoader


SEED = int(os.getenv("SEED", 123))
torch.manual_seed(SEED)


def test_vit() -> None:
    # vit-base-patch16-224", cache_dir="/data/vit_model")
    model = ViTForImageClassification.from_pretrained("/data/vit")

    device = torch.device("cuda")
    model = model.to(device=device)
    model.train()

    batch_size = 5
    dataset = ViTImageDataset(type="eval")
    train_dataloader = DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  prefetch_factor=2,
                                  num_workers=multiprocessing.cpu_count(),
                                  pin_memory=True if device.type == 'cuda' else False)
    i = 0
    image_batch_features: BatchFeature
    label_info_batch: LabelType
    for image_batch_features, label_info_batch in train_dataloader:
        if i >= 5:
            break
        batched_pixel_values = image_batch_features.pixel_values[0].to(device)
        outputs = model(pixel_values=batched_pixel_values)

        logits = outputs.logits
        predicted_class_indices = logits.argmax(-1)

        actual_class_ids_batch = label_info_batch["cls"].squeeze().to(device)

        for k_in_batch in range(batch_size):
            pred_idx = predicted_class_indices[k_in_batch].item()
            actual_cls_id = actual_class_ids_batch[k_in_batch].item()

            # predicted_label_str = model.config.id2label.get(pred_idx, f"Unknown ID: {pred_idx}")
            # actual_label_str = model.config.id2label.get(actual_cls_id, f"Unknown ID: {actual_cls_id}")

            print(
                f"[Batch {i}:{k_in_batch}] Predicted: {pred_idx}, Actual: {actual_cls_id}"
            )
        i += 1
