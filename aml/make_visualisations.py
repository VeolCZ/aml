import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def add_supper_classes_to_data(data: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Take the dataset add a column with the supper-species and
    save the new dataset (if save is set to True).

    Args:
        data(pd.DataFrame): dataset to which to add the column
        save(bool): boolean representing whether or not to save the new dataset

    Returns:
        pd.DataFrame: the new dataframe
    """
    super_classes: dict[str, list[int]] = {}

    with open("/data/CUB_200_2011/classes.txt", "r", encoding="utf-8") as file:
        for line in file:
            index, name = line.split()
            name = name.split(sep="_")[-1]
            if name in super_classes:
                super_classes[name].append(int(index))
            else:
                super_classes[name] = [int(index)]

    data["super_species"] = "unnamed"

    for key, species in super_classes.items():
        data.loc[data["class_id"].isin(species), "super_species"] = key
    if save:
        data.to_csv("/data/labels_with_superspecies.csv", index=False)
    return data


def produce_tsne(data: pd.DataFrame) -> None:
    """
    Apply PCA on the dataset, then produce a t-SNE plot and and
    save it.

    Args:
        data(pd.DataFrame): the dataframe in question

    Returns:
        None
    """
    y = data[["class_id", "x", "y", "width", "height", "super_species"]]

    cols = data.columns.tolist()
    start = cols.index("attr_1_pres")
    end = cols.index("attr_312_pres") + 1
    x = data.iloc[:, start:end]

    tsne = TSNE(n_components=2, random_state=42)
    pca = PCA(n_components=0.6)
    x_pca = pca.fit_transform(x)
    x_embedded = tsne.fit_transform(x_pca)

    tsne_df = pd.DataFrame(x_embedded, columns=["TSNE1", "TSNE2"])
    tsne_df[["class_id", "x", "y", "width", "height", "super_species"]] = y.values

    plt.figure(figsize=(30, 20))
    sns.scatterplot(
        data=tsne_df,
        x="TSNE1",
        y="TSNE2",
        hue="super_species",
        style="super_species",
        palette="tab10",
        alpha=0.7,
    )
    plt.title("t-SNE Plot")
    plt.legend(title="Label", ncol=2)
    plt.savefig("/logs/tsne.jpg")
    plt.close()


def produce_class_freq_histogram(data: pd.DataFrame) -> None:
    """
    Produce a histogram of class frequencies for the dataframe and save it.

    Args:
        data(pd.DataFrame): the dataframe in question

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    plt.hist(data["class_id"], bins=200)
    plt.xlabel("class_id")
    plt.ylabel("frequency")
    plt.savefig("/logs/freq_hist.jpg")
    plt.close()


def produce_certainty_plot(data: pd.DataFrame) -> None:
    """
    Produce a barplot of the mean certainty of the binary operators and save it.

    Args:
        data(pd.DataFrame): the dataframe in question

    Returns:
        None
    """
    first_cert = data.columns.get_loc("attr_1_cert")
    cert_sums_averages = data.iloc[:, first_cert:-1].sum() / data.shape[0]
    cert_sums_averages.plot(kind="bar", color="skyblue")
    plt.xticks(
        ticks=list(range(0, len(cert_sums_averages), 10)),
        labels=[str(i) for i in range(0, len(cert_sums_averages), 10)],
    )
    plt.xlabel("Attribute index")
    plt.ylabel("Mean certainty")
    plt.savefig("/logs/certanty_plot.jpg")
    plt.close()


def make_visualization() -> None:
    data_birds = pd.read_csv("/data/labels.csv")
    data_birds = add_supper_classes_to_data(data_birds, save=False)
    produce_tsne(data_birds.copy())
    produce_class_freq_histogram(data_birds.copy())
    produce_certainty_plot(data_birds.copy())
