from project.algorithms.algorithms import *
from project.data_loader.data_loader import load_fmnist, load_reuters, load_smallnorb


def set_up_logger():
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')


def run_algorithms(dataset_loader, dataset_name: str):
    logging.info(dataset_name + " dataset is fetching.")
    x, y = dataset_loader()
    logging.info("Dataset was fetched to memory.")
    execute_pacmap(x, y, dataset_name)
    execute_trimap(x, y, dataset_name)


if __name__ == '__main__':
    set_up_logger()
    run_algorithms(load_fmnist, "FMnist")
    run_algorithms(load_reuters, "Reuters")
    run_algorithms(load_smallnorb, "Smallnorb")
