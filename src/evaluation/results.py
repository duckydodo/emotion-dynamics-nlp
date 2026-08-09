import json
from datetime import datetime
from pathlib import Path
import subprocess

from src.config import OUTPUT_DIR
from src.data.types import EvaluationResult


EXPERIMENTS_DIR = OUTPUT_DIR / "experiments"


def _experiment_dir(experiment_name: str) -> Path:
    """
    Create (if necessary) and return the directory
    for one experiment.
    """
    path = EXPERIMENTS_DIR / experiment_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_metrics(
    results: EvaluationResult,
    experiment_name: str,
) -> None:
    """
    Save evaluation metrics to metrics.json
    """

    output_dir = _experiment_dir(experiment_name)

    metrics = {
        "accuracy": results.accuracy,
        "macro_f1": results.macro_f1,
        "weighted_f1": results.weighted_f1,
        "per_class": {},
    }

    for label, values in results.per_class.items():
        metrics["per_class"][label] = {
            "precision": values.precision,
            "recall": values.recall,
            "f1": values.f1,
        }

    with open(output_dir / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)


def save_metadata(
    experiment_name: str,
    **metadata,
) -> None:
    """
    Save experiment configuration.
    """

    output_dir = _experiment_dir(experiment_name)

    metadata["timestamp"] = datetime.now().isoformat()

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()

        metadata["git_commit"] = commit

    except Exception:
        metadata["git_commit"] = None

    with open(output_dir / "metadata.json", "w") as file:
        json.dump(metadata, file, indent=4)


def load_metrics(
    experiment_name: str,
) -> dict:

    path = (
        _experiment_dir(experiment_name)
        / "metrics.json"
    )

    with open(path) as file:
        return json.load(file)


def load_metadata(
    experiment_name: str,
) -> dict:

    path = (
        _experiment_dir(experiment_name)
        / "metadata.json"
    )

    with open(path) as file:
        return json.load(file)
        
        
from src.evaluation.metrics import plot_confusion_matrix


def save_confusion_matrix(
    results: EvaluationResult,
    experiment_name: str,
) -> None:

    output_dir = _experiment_dir(experiment_name)

    plot_confusion_matrix(
        results,
        save_path=output_dir / "confusion_matrix.png",
        show=False,
    )
