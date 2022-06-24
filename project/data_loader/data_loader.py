from typing import Tuple

import pandas as pd
from pandas import DataFrame

FMNIST = 'resources/datasets/fmnist.csv'
REUTERS = 'resources/datasets/reuters.csv'
SMALLNORB = 'resources/datasets/smallnorb.csv'


def __load_dataset(path: str, target_idx=-1, drop=None) -> Tuple[DataFrame, DataFrame]:
    dataset = pd.read_csv(path, header=None)
    if drop is not None:
        dataset = dataset[dataset.iloc[:, target_idx] <= drop]

    dataset = get_part_of_dataset(dataset)
    x = dataset.drop(columns=dataset.columns[target_idx])
    y = dataset.iloc[:, target_idx]
    return x, y


def get_part_of_dataset(dataset, percent=10):
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    dataset = dataset.head(int(len(dataset) * (percent / 100)))
    return dataset


def load_fmnist() -> Tuple[DataFrame, DataFrame]:
    return __load_dataset(FMNIST)


def load_reuters() -> Tuple[DataFrame, DataFrame]:
    return __load_dataset(REUTERS, drop=8)


def load_smallnorb() -> Tuple[DataFrame, DataFrame]:
    return __load_dataset(SMALLNORB)
