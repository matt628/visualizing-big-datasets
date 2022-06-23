from typing import Dict

from numpy.typing import ArrayLike

from project.metrics.lab_metrics import LocalMetric

RESOURCE_PATH = 'resources/plots/'


def visualize_metric(x_results: Dict[str, ArrayLike], x: ArrayLike, y: ArrayLike, dataset: str):
    local_metrics = LocalMetric(RESOURCE_PATH + dataset + '/')
    for key, transform_x in x_results.items():
        local_metrics.calculate_knn_gain_and_dr_quality(
            X_lds=transform_x,
            X_hds=x,
            labels=y.astype(int),
            method_name=dataset + " & " + key)
    local_metrics.visualize()
