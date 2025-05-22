from matplotlib import pyplot as plt
import seaborn as sns
import io


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