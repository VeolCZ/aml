from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from numpy.typing import NDArray
from preprocessing.TreeImageDataset import TreeImageDataset
from torch.utils.data import DataLoader
import torch

def compute_many_iou(many_boxes1, many_boxes2):
    i=0
    sum=0
    for box1, box2 in zip(many_boxes1, many_boxes2):
        sum+= compute_iou(box1, box2)
        i+=1
    return sum/i

def compute_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    if union_area > 0:
        iou_tensor = inter_area / union_area
        return iou_tensor.item()
    else:
        return 0

def train_random_forest(
    x: NDArray[np.float64], y, test_size: float = 0.2
) -> tuple[RandomForestRegressor, float]:
    # Flattening
    x = [sample.flatten() for sample in x]
    x = np.array(x, dtype=np.float64)

    forest_regressor = RandomForestRegressor()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        [label["bbox"] for label in y],
        test_size=test_size,
        stratify=[label["cls"] for label in y]
    )

    print("Fitting has begun... dun dun dunnn...")
    y_train = [y.squeeze() for y in y_train]
    y_test = [y.squeeze() for y in y_test]
    forest_regressor.fit(x_train, y_train)
    print(y_train)
    print("predicting, hopefully better than my future")
    y_pred = forest_regressor.predict(x_test)

    print("figuring out if this thing is better than random guessing")
    iou = compute_many_iou(y_test, y_pred)
    print("IOU: ", iou)

    return forest_regressor, iou


def my_collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)


def treereg_test():
    train_dataloader = DataLoader(TreeImageDataset("train"), collate_fn=my_collate_fn)
    all_x = []
    all_y = []
    idx = 0

    for x_batch, y_batch in train_dataloader:
        print(f"Batch {idx}")
        idx += 1
        if idx >= 21:
            print("Karolina's computer likes being alive")
            break

        # x_batch_np = np.array([x for x in x_batch], dtype=np.float64)
        # y_batch_np = [int(label["bbox"].view(-1)[0].item()) for label in y_batch]
        all_x.extend(x_batch)
        all_y.extend(y_batch)

    #all_x = np.vstack(all_x)
    #all_y = np.array(all_y, dtype=np.int64)
    train_random_forest(all_x, all_y)
