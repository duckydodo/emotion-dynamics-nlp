from pathlib import Path

import pandas as pd

from src.evaluation.results import load_metrics


def compare_experiments(
    experiment_names: list[str],
) -> pd.DataFrame:
    """
    Compare saved experiments using their main evaluation metrics.

    Returns a DataFrame sorted by Macro F1, highest first.
    """

    rows = []

    for experiment_name in experiment_names:
        metrics = load_metrics(experiment_name)

        rows.append(
            {
                "Model": experiment_name,
                "Accuracy": metrics["accuracy"],
                "Macro F1": metrics["macro_f1"],
                "Weighted F1": metrics["weighted_f1"],
            }
        )

    comparison = pd.DataFrame(rows)

    return comparison.sort_values(
        "Macro F1",
        ascending=False,
    ).reset_index(drop=True)
