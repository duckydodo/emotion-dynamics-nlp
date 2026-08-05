from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score,
)

from src.data.types import EvaluationResult, ClassMetrics

def evaluate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
    )

    labels = sorted(y_true.unique())
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    per_class = {}

    for label, p, r, f in zip(labels, precision, recall, f1, strict=True):
        per_class[label] = ClassMetrics(
            precision=p,
            recall=r,
            f1=f,
        )
    
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )
    
    return EvaluationResult(
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,

        per_class=per_class,

        confusion_matrix=cm,
    )
    
    
def print_results(results: EvaluationResult) -> None:

    print("\n===== EVALUATION =====")

    print(f"Accuracy     : {results.accuracy:.4f}")
    print(f"Macro F1     : {results.macro_f1:.4f}")
    print(f"Weighted F1  : {results.weighted_f1:.4f}")

    print("\nPer-class metrics")
    print("-" * 50)
    print(f"{'Emotion':<12}{'Precision':>12}{'Recall':>12}{'F1':>12}")

    for label, metrics in results.per_class.items():
        print(
            f"{label:<12}"
            f"{metrics.precision:>12.4f}"
            f"{metrics.recall:>12.4f}"
            f"{metrics.f1:>12.4f}"
        )

    print("\nConfusion Matrix")
    print(results.confusion_matrix)
    
    
import matplotlib.pyplot as plt


def plot_confusion_matrix(results, save_path=None):

    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(results.confusion_matrix)

    ax.set_xticks(range(len(results.per_class)))
    ax.set_yticks(range(len(results.per_class)))

    labels = list(results.per_class.keys())

    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(results.confusion_matrix.shape[0]):
        for j in range(results.confusion_matrix.shape[1]):
            ax.text(
                j,
                i,
                str(results.confusion_matrix[i, j]),
                ha="center",
                va="center",
            )

    fig.colorbar(im)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()
