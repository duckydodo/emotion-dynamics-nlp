from fastapi.responses import FileResponse
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.application.analyzer import ConversationAnalyzer, MODEL_METADATA
from src.models.inference import CHECKPOINTS


app = FastAPI(
    title="Emotion Dynamics NLP API",
    description="Context-aware emotion analysis for multi-turn conversations.",
    version="1.0.0",
)

analyzer = ConversationAnalyzer()

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path("static/index.html"))

class ConversationRequest(BaseModel):
    conversation: str = Field(
        ...,
        description="Conversation using one Speaker: utterance per turn.",
    )
    mode: str = Field(
        default="utterance",
        description="Model mode: utterance, context_1, context_1_speaker, or dialoguernn.",
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "available_modes": list(CHECKPOINTS.keys()),
    }


@app.get("/models")
def list_models():
    """
    Returns the canonical model registry containing verified metadata for the
    three canonical model families: DistilBERT, DialogueRNN, and DialogueGCN.
    """
    return analyzer.get_model_registry()


@app.get("/benchmarks")
def get_benchmarks():
    """
    Returns verified research benchmarks on Canonical MELD Cohort A (N=2,328),
    including overall evaluation metrics, per-class performance, statistical tests,
    and confusion matrices across the canonical model families.
    """
    import json
    import csv
    from pathlib import Path

    overall_metrics = []
    metrics_path = Path("outputs/tables/dialoguegcn_cohort_a_metrics.csv")
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                overall_metrics.append({
                    "model": row["Model"],
                    "parameters": int(row["Parameters"]),
                    "receptive_field": row["Receptive_Field"],
                    "accuracy": float(row["Cohort_A_Accuracy"]),
                    "macro_f1": float(row["Cohort_A_Macro_F1"]),
                    "weighted_f1": float(row["Cohort_A_Weighted_F1"]),
                })

    per_class = {"distilbert": [], "dialoguernn": [], "dialoguegcn": []}
    drnn_pc_path = Path("outputs/tables/dialoguernn_per_class.csv")
    if drnn_pc_path.exists():
        with open(drnn_pc_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Cohort") == "Cohort A: Strict Contiguous Matched Subset":
                    model_name = row.get("Model", "")
                    target_key = None
                    if "DistilBERT Utterance Baseline" in model_name:
                        target_key = "distilbert"
                    elif "DialogueRNN (Unweighted)" in model_name:
                        target_key = "dialoguernn"
                    if target_key:
                        per_class[target_key].append({
                            "emotion": row["Emotion"],
                            "precision": round(float(row["Precision"]), 4),
                            "recall": round(float(row["Recall"]), 4),
                            "f1": round(float(row["F1"]), 4),
                            "support": int(row["Support"]),
                        })

    dgcn_pc_path = Path("outputs/tables/dialoguegcn_per_class.csv")
    if dgcn_pc_path.exists():
        with open(dgcn_pc_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                per_class["dialoguegcn"].append({
                    "emotion": row["Emotion"],
                    "precision": round(float(row["Precision"]), 4),
                    "recall": round(float(row["Recall"]), 4),
                    "f1": round(float(row["F1"]), 4),
                    "support": int(row["Support"]),
                })

    labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    confusion_matrices = {}

    cm_path = Path("outputs/tables/canonical_confusion_matrices.json")
    if cm_path.exists():
        with open(cm_path, "r", encoding="utf-8") as f:
            cms = json.load(f)
            for k, matrix in cms.items():
                if "DistilBERT Utterance Baseline" in k and "Cohort A" in k:
                    confusion_matrices["distilbert"] = matrix

    drnn_cm_path = Path("outputs/experiments/dialoguernn_unweighted/confusion_matrix.json")
    if drnn_cm_path.exists():
        with open(drnn_cm_path, "r", encoding="utf-8") as f:
            confusion_matrices["dialoguernn"] = json.load(f)

    dgcn_cm_path = Path("outputs/tables/dialoguegcn_confusion_matrix.json")
    if dgcn_cm_path.exists():
        with open(dgcn_cm_path, "r", encoding="utf-8") as f:
            cms = json.load(f)
            if "Cohort A" in cms:
                confusion_matrices["dialoguegcn"] = cms["Cohort A"]

    sig_path = Path("outputs/tables/dialoguegcn_statistical_tests.csv")
    statistical_tests = []
    if sig_path.exists():
        with open(sig_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                statistical_tests.append({
                    "comparison": row["Comparison"],
                    "n": int(row["N"]),
                    "dialoguegcn_macro_f1": round(float(row["DialogueGCN_Macro_F1"]), 4),
                    "benchmark_macro_f1": round(float(row["Benchmark_Macro_F1"]), 4),
                    "delta_macro_f1": round(float(row["Delta_Macro_F1"]), 4),
                    "ci_lower": round(float(row["Bootstrap_CI_Lower"]), 4),
                    "ci_upper": round(float(row["Bootstrap_CI_Upper"]), 4),
                    "permutation_p_value": float(row["Permutation_p_value"]),
                    "mcnemar_p_value": float(row["McNemar_p"]),
                })

    return {
        "cohort": "Cohort A: Strict Contiguous Matched Subset (N=2,328)",
        "labels": labels,
        "overall_metrics": overall_metrics,
        "per_class": per_class,
        "confusion_matrices": confusion_matrices,
        "statistical_tests": statistical_tests,
    }


MAX_PAYLOAD_BYTES: int = 10 * 1024 * 1024  # 10 MB defensive bound


def validate_payload_size(raw_text: str) -> None:
    if len(raw_text.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Payload exceeds maximum permitted size of {MAX_PAYLOAD_BYTES // (1024 * 1024)} MB.",
        )


@app.post("/analyze")
def analyze(request: ConversationRequest):
    validate_payload_size(request.conversation)
    try:
        return analyzer.analyze(
            conversation=request.conversation,
            mode=request.mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Required model checkpoint is unavailable.")


@app.post("/predict")
def predict(request: ConversationRequest):
    validate_payload_size(request.conversation)
    try:
        turns = analyzer.parse_conversation(request.conversation)
        model = analyzer._get_model(request.mode)
        predictions = model.predict(turns)

        return {
            "mode": request.mode,
            "model_metadata": MODEL_METADATA.get(request.mode, {}),
            "turns": predictions,
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Required model checkpoint is unavailable.")


class CompareRequest(BaseModel):
    conversation: str = Field(
        ...,
        description="Conversation text or export to compare across model families.",
    )
    models: Optional[list[str]] = Field(
        default=None,
        description="Optional list of model IDs to compare (e.g. ['distilbert', 'dialoguernn', 'dialoguegcn']).",
    )


@app.post("/compare")
def compare_models(request: CompareRequest):
    validate_payload_size(request.conversation)
    try:
        return analyzer.compare(
            conversation=request.conversation,
            model_ids=request.models,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Required model checkpoint is unavailable.")


class ImportRequest(BaseModel):
    conversation: str = Field(
        ...,
        description="Conversation text or export content to parse and normalize.",
    )
    format: str = Field(
        default="auto",
        description="Source format hint: auto, plain_text, instagram_json, csv, etc.",
    )
    column_mapping: Optional[dict[str, str]] = Field(
        default=None,
        description="Optional column mapping for CSV imports (e.g. {'speaker': 'Speaker', 'text': 'Utterance'}).",
    )


@app.post("/import")
def import_conversation(request: ImportRequest):
    validate_payload_size(request.conversation)
    try:
        canonical = analyzer.parse_canonical(
            request.conversation,
            format_hint=request.format,
            column_mapping=request.column_mapping,
        )
        val_errors = canonical.validate()
        evaluable_turns = canonical.get_evaluable_turns()

        # Date range calculation
        timestamps = [m.timestamp for m in canonical.messages if m.timestamp is not None]
        date_range = None
        if timestamps:
            min_ts = min(timestamps)
            max_ts = max(timestamps)
            date_range = {
                "start": min_ts.isoformat(),
                "end": max_ts.isoformat(),
                "duration_seconds": (max_ts - min_ts).total_seconds(),
            }

        unsupported_count = len([m for m in canonical.messages if not m.is_evaluable])

        return {
            "status": "warning" if (val_errors or canonical.warnings) else "success",
            "conversation_id": canonical.conversation_id,
            "source_format": canonical.source_format.value,
            "num_messages": len(canonical.messages),
            "parsed_message_count": len(canonical.messages),
            "evaluable_turn_count": len(evaluable_turns),
            "num_evaluable_turns": len(evaluable_turns),
            "num_speakers": len(canonical.participants),
            "participant_count": len(canonical.participants),
            "speakers": [p.name for p in canonical.participants],
            "unsupported_message_count": unsupported_count,
            "warning_count": len(canonical.warnings),
            "warnings": [w.to_dict() for w in canonical.warnings],
            "validation_errors": val_errors,
            "date_range": date_range,
            "preview_messages": [m.to_dict() for m in canonical.messages[:5]],
            "normalized_conversation": canonical.to_dict(),
            "import_metadata": {
                **canonical.metadata,
                "has_timestamps": canonical.metadata.get("has_timestamps", False),
            },
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
