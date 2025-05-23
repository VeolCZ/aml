from matplotlib import pyplot as plt
import seaborn as sns
import io
import torch
from PIL import Image
import numpy as np


def plot_confusion_matrix(confusion_matrix: torch.Tensor, cls: int) -> io.BytesIO:
    class_names = [x for x in range(1, cls+1)]
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    image = Image.open(buf).convert("RGB")
    image = np.array(image).astype(np.float32)/255.0
    image = torch.tensor(image).permute(2, 0, 1)
    return image
