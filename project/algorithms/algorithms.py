import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pacmap
import trimap
import umap
import numpy as np
from numpy.typing import ArrayLike
from sklearn.manifold import LocallyLinearEmbedding

pacmap_args = {
    'fmnist': {
        'n_components': 2,
        'n_neighbors': 8,
        'MN_ratio': 1.5,
        'distance': 'euclidean',
        'random_state': 45
    }, "smallnorb": {
        'n_components': 2,
        'n_neighbors': 12,
        'FP_ratio': 0.2,
        'MN_ratio': 3.0,
        'distance': 'angular',
        'num_iters': 1000,
        'random_state': 45
    }, 'reuters': {
        'n_components': 2,
        'n_neighbors': 10,
        'MN_ratio': 1.5,
        'distance': 'angular',
        'random_state': 45
    }
}

trimap_args = {
    'fmnist': {
        'n_dims': 2,
        'distance': 'euclidean'
    }, "smallnorb": {
        'n_dims': 2,
        'n_inliers': 18,
        'n_outliers': 6,
        'n_random': 5,
        'distance': 'cosine'
    }, 'reuters': {
        'n_dims': 2,
        'distance': 'cosine'
    }
}

umap_args = {
    'fmnist': {
        'n_components': 2,
        'n_neighbors': 8,
        'spread': 1.4,
        'metric': 'euclidean',
        'random_state': 45
    }, "smallnorb": {
        'n_components': 2,
        'n_neighbors': 12,
        'spread': 0.5,
        'metric': 'cosine',
        'random_state': 45
    }, 'reuters': {
        'n_components': 2,
        'n_neighbors': 10,
        'spread': 0.4,
        'metric': 'cosine',
        'random_state': 45
    },
}

lle_args = {
    'fmnist': {
        "eigen_solver": "auto",
        'n_components': 2,
        'n_neighbors': 8,
        'random_state': 45
    }, "smallnorb": {
        'n_components': 2,
        'n_neighbors': 12,
        'random_state': 45
    }, 'reuters': {
        'n_components': 2,
        'n_neighbors': 10,
        'random_state': 45
    },
}


def execute_pacmap(x: ArrayLike, y: ArrayLike, dataset_name: str):
    embedding = pacmap.PaCMAP(**pacmap_args.get(dataset_name))
    logging.info("PaCMAP has been launched.")
    x_transformed = embedding.fit_transform(x)
    logging.info("PaCMAP was completed.")
    __show_result(x_transformed, y, dataset_name, 'pacmap')
    return x_transformed


def execute_trimap(x: ArrayLike, y: ArrayLike, dataset_name: str):
    embedding = trimap.TRIMAP(**trimap_args.get(dataset_name))
    logging.info("TRIMAP has been launched.")
    x_transformed = embedding.fit_transform(x)
    logging.info("TRIMAP was completed.")
    __show_result(x_transformed, y, dataset_name, 'trimap')
    return x_transformed


def execute_umap(x: ArrayLike, y: ArrayLike, dataset_name: str):
    embedding = umap.UMAP(**umap_args.get(dataset_name))
    logging.info("UMAP has been launched.")
    x_transformed = embedding.fit_transform(x)
    logging.info("UMAP was completed.")
    __show_result(x_transformed, y, dataset_name, 'umap')
    return x_transformed


def execute_lle(x: ArrayLike, y: ArrayLike, dataset_name: str):
    embedding = LocallyLinearEmbedding(**lle_args.get(dataset_name))
    logging.info("LLE has been launched.")
    x_transformed = embedding.fit_transform(x)
    logging.info("LLE was completed.")
    __show_result(x_transformed, y, dataset_name, 'lle')
    return x_transformed


def __show_result(x_transformed: ArrayLike, y: ArrayLike, dataset: str, algo: str):
    plt.figure().clear()
    df = pd.DataFrame()
    df["y"] = y
    df["Component1"] = x_transformed[:, 0]
    df["Component2"] = x_transformed[:, 1]
    chart = sns.scatterplot(x="Component1",
                            y="Component2",
                            hue=y.tolist(),
                            palette=sns.color_palette("bright", __number_of_categories(y)),
                            data=df)
    # chart.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend([], [], frameon=False)
    chart.set(title=dataset + ' & ' + algo)
    plt.savefig('resources/plots/' + dataset + '/' + algo)
    plt.show()


def __number_of_categories(y):
    return len(set(y))
