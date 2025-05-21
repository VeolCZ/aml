from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import seaborn as sns
import io
from ViT.ViT import epoch_loss

def plot_confusion_matrix(confusion_matrix, cls):
    class_names = [x for x in range(1, cls+1)]
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf



"""
Access eval metrics an somehow extract them from dict or another method
Graphs:
- matrix for forests
- matrix for trees
- loss plot for vit
- time for inference
maybe plots:
- accuracy with num of images
- iou with num of images
- f1 with num of images
- hyperparams for ViT (maybe)
- nice display of all eval metrics instead of the current implementation
"""

writer = SummaryWriter()
for epoch in epoch_loss.keys():
    writer.addscalar("training validation",epoch_loss[epoch],epoch)
