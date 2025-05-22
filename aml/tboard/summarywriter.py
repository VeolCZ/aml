from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from tboard.plotting import plot_confusion_matrix
from random_forests.forest_train_util import train_classifier_forest  # , train_regressor_forest
# from ViT.ViT_utils import train_vit
# rom ViT.ViT import epoch_loss
# from evaluator.Evaluator import Evaluator

forest_cls_eval = train_classifier_forest()
# orest_reg_eval = train_regressor_forest() # comment it out if only classifier is run
# orest_cls_eval["iou"] = forest_reg_eval # same here
# iT_eval = train_vit()
confusion_matrix = eval["confusion_matrix"]
num_classes = eval["num_classes"]
buf = plot_confusion_matrix(confusion_matrix.cpu().numpy(), num_classes)
image = plt.imread(buf, format='png')
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
# r epoch in epoch_loss.keys():
# riter.addscalar("training validation",epoch_loss[epoch],epoch)
writer.add_image("Confusion_Matrix", image, global_step=0, dataformats='HWC')
writer.close()
