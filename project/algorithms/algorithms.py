import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pacmap
import trimap
from numpy.typing import ArrayLike
from sklearn.manifold import Isomap


def execute_pacmap(x: ArrayLike, y: ArrayLike, dataset_name: str):
    embedding = pacmap.PaCMAP(n_components=2)
    logging.info("PaCMAP has been launched.")
    x_transformed = embedding.fit_transform(x)
    logging.info("PaCMAP was completed.")
    __show_result(x_transformed, y, "PaCMAP & " + dataset_name)


def execute_trimap(x: ArrayLike, y: ArrayLike, dataset_name: str):
    embedding = trimap.TRIMAP(n_dims=2)
    logging.info("TRIMAP has been launched.")
    x_transformed = embedding.fit_transform(x)
    logging.info("TRIMAP was completed.")
    __show_result(x_transformed, y, "TRIMAP & " + dataset_name)


def execute_isomap(x: ArrayLike, y: ArrayLike, dataset_name: str):
    embedding = Isomap(n_components=2)
    logging.info("Isomap has been launched.")
    x_transformed = embedding.fit_transform(x)
    logging.info("Isomap was completed.")
    __show_result(x_transformed, y, "Isomap & " + dataset_name)


def __show_result(x_transformed: ArrayLike, y: ArrayLike, chart_title: str):
    df = pd.DataFrame()
    df["y"] = y
    df["Component1"] = x_transformed[:, 0]
    df["Component2"] = x_transformed[:, 1]
    chart = sns.scatterplot(x="Component1",
                            y="Component2",
                            hue=y.tolist(),
                            palette=sns.color_palette("hls", __number_of_categories(y)),
                            data=df)
    chart.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    chart.set(title=chart_title)
    plt.show()


def __number_of_categories(y):
    return len(set(y))
