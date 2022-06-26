import logging
from typing import Callable
from project.algorithms.algorithms import *
from project.data_loader.data_loader import load_fmnist, load_reuters, load_smallnorb
from project.metrics.metrics_visalizer import visualize_metric


def set_up_logger():
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename='logsReuters.log',
                        filemode='w')


def run_algorithms(dataset_loader: Callable, dataset_name: str):
    logging.info(dataset_name + " processing has been started.")

    logging.info("Dataset is fetching.")
    x, y = dataset_loader()
    x, y = x.to_numpy(), y.to_numpy()
    logging.info("Dataset was fetched to memory.")

    logging.info("Run embedding algorithms.")
    embeddings = {
        'pacmap': execute_pacmap(x, y, dataset_name),
        'trimap': execute_trimap(x, y, dataset_name),
        'umap': execute_umap(x, y, dataset_name),
        # 'lle': execute_lle(x, y, dataset_name)
    }
    logging.info("Processing was finished.")

    logging.info("Run method to calculate metrics.")
    visualize_metric(embeddings, x, y, dataset_name)
    logging.info("Metrics were calculated.")

    logging.info(dataset_name + " processing is completed.")


if __name__ == '__main__':
    set_up_logger()
    # run_algorithms(load_fmnist, "fmnist")
    run_algorithms(load_reuters, "reuters")
    # run_algorithms(load_smallnorb, "smallnorb")
