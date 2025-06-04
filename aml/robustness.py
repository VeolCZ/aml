import albumentations as A  #wowow
from albumentations.pytorch import ToTensorV2
from ViT.ViT_utils import eval_vit
import Union
import os


SEED = int(os.getenv("SEED", "123"))


def calculate_robustness(severity_min = 0, severity_max = 1):
    pass
            
            



