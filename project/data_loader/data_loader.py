from typing import Tuple

import pandas as pd
from pandas import DataFrame

FMNIST = 'resources/fmnist.csv'
REUTERS = 'resources/reuters.csv'
SMALLNORB = 'resources/smallnorb.csv'


def __load_dataset(path: str, target_idx=-1) -> Tuple[DataFrame, DataFrame]:
    dataset = pd.read_csv(path, header=None)
    x = dataset.drop(columns=dataset.columns[target_idx])
    y = dataset.iloc[:, target_idx]
    return x, y


def load_fmnist() -> Tuple[DataFrame, DataFrame]:
    return __load_dataset(FMNIST)


def load_reuters() -> Tuple[DataFrame, DataFrame]:
    return __load_dataset(REUTERS)


def load_smallnorb() -> Tuple[DataFrame, DataFrame]:
    return __load_dataset(SMALLNORB)
