from fastapi.responses import FileResponse
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.application.analyzer import ConversationAnalyzer
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
        description="Model mode: utterance, context_1, or context_1_speaker.",
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "available_modes": list(CHECKPOINTS.keys()),
    }


@app.post("/analyze")
def analyze(request: ConversationRequest):
    try:
        return analyzer.analyze(
            conversation=request.conversation,
            mode=request.mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict")
def predict(request: ConversationRequest):
    try:
        turns = analyzer.parse_conversation(request.conversation)
        model = analyzer._get_model(request.mode)
        predictions = model.predict(turns)

        return {
            "mode": request.mode,
            "turns": predictions,
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
