import os
import pandas as pd


def parse_labels(base_path: str) -> pd.DataFrame:
    """
    Parses the dataset labels and loads it to a dataframe.

    Args:
        base_path (str): Path to the directory containing the dataset

    Returns:
        pd.DataFrame: Dataset with one row per image
    """

    images = pd.read_csv(os.path.join(base_path, "images.txt"),
                         sep=" ", header=None, names=["image_id", "image_name"])
    train_test = pd.read_csv(os.path.join(base_path, "train_test_split.txt"),
                             sep=" ", header=None, names=["image_id", "is_training"])
    image_labels = pd.read_csv(os.path.join(base_path, "image_class_labels.txt"),
                               sep=" ", header=None, names=["image_id", "class_id"])
    bounding_boxes = pd.read_csv(os.path.join(base_path, "bounding_boxes.txt"),
                                 sep=" ", header=None, names=["image_id", "x", "y", "width", "height"])
    attribute_labels_names = ["image_id", "attribute_id", "is_present", "certainty_id", "time"]
    attribute_labels = pd.read_csv(os.path.join(base_path, "attributes", "image_attribute_labels_clean.txt"),
                                   sep=" ", header=None,
                                   names=attribute_labels_names)

    image_attributes = attribute_labels.groupby(["image_id", "attribute_id"])

    attr_presence = image_attributes["is_present"].mean().unstack()
    attr_presence.columns = [f"attr_{col}_pres" for col in attr_presence.columns]

    attr_certainty = image_attributes["certainty_id"].mean().unstack()
    attr_certainty.columns = [f"attr_{col}_cert" for col in attr_certainty.columns]

    image_attributes_df = attr_presence.merge(attr_certainty, left_index=True, right_index=True, how="left")

    df = images.merge(train_test, on="image_id")
    df = df.merge(image_labels, on="image_id")
    df = df.merge(bounding_boxes, on="image_id")
    df = df.merge(image_attributes_df, on="image_id", how="left")
    df["image_path"] = base_path + "/images/" + df["image_name"]

    columns = ["image_id", "image_path", "class_id", "is_training", "x", "y", "width", "height"]
    columns += [col for col in df.columns if col.startswith("attr_") and col.endswith("pres")]
    columns += [col for col in df.columns if col.startswith("attr_") and col.endswith("cert")]

    final_df = df[columns].copy()

    corrupt_mask = final_df.index.isin([447, 1400, 3616, 3618, 3779, 5028, 5392, 6320])
    final_df = final_df[~corrupt_mask]

    return final_df


def make_labels() -> None:
    """
    Creates and saves a labels.csv file in the /data directory.
    """
    assert os.path.exists("/data/CUB_200_2011"), "Please ensure the dataset is properly extracted into /data"

    df = parse_labels("/data/CUB_200_2011")
    df.to_csv("/data/labels.csv", index=False)


def get_classes() -> dict[int, str]:
    """
    Returns a dictionary mapping class IDs to class names.
    """
    assert os.path.exists("/data/CUB_200_2011/classes.txt"), "Please ensure the dataset is properly \
         extracted into /data"

    df = pd.read_csv("/data/CUB_200_2011/classes.txt", sep=" ", header=None, names=["id", "class"])
    df = df["class"].apply(lambda x: x.split(".")[1].replace("_", " "))
    images: dict[int, str] = df.to_dict()
    return images
