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

        acc = metrics.get("accuracy", metrics.get("test_accuracy"))
        macro = metrics.get("macro_f1", metrics.get("test_macro_f1"))
        weighted = metrics.get("weighted_f1", metrics.get("test_weighted_f1"))
        examples = metrics.get("test_examples", None)

        rows.append(
            {
                "Model": experiment_name,
                "Examples": examples,
                "Accuracy": acc,
                "Macro F1": macro,
                "Weighted F1": weighted,
            }
        )

    comparison = pd.DataFrame(rows)

    return comparison.sort_values(
        "Macro F1",
        ascending=False,
    ).reset_index(drop=True)
