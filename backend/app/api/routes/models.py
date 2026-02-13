"""
Models API routes - list available models
"""
from fastapi import APIRouter

from app.core.config import settings
from app.services.model_service import model_service
from app.schemas.analysis import ModelsResponse, ModelInfo

router = APIRouter()

# Evaluation metrics from TASK-005-EVALUATION-EN (test set results)
MODEL_METRICS = {
    "distilbert": {
        "accuracy": 0.9489,
        "precision": 0.5992,
        "recall": 0.5932,
        "f1": 0.5884,
        "parameters": "66M",
        "architecture": "Transformer",
        "training_time": "25.9 мин",
        "training_config": {
            "epochs": 6,
            "learning_rate": "3e-5",
            "batch_size": 32,
            "max_length": 128,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "dropout": 0.1,
            "dataset": "UD English EWT + GUM (~489k tokens, 406 labels)",
        },
    },
    "bert": {
        "accuracy": 0.9483,
        "precision": 0.5783,
        "recall": 0.5731,
        "f1": 0.5679,
        "parameters": "109M",
        "architecture": "Transformer",
        "training_time": "38.6 мин",
        "training_config": {
            "epochs": 5,
            "learning_rate": "2e-5",
            "batch_size": 32,
            "max_length": 128,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "dropout": 0.1,
            "dataset": "UD English EWT + GUM (~489k tokens, 406 labels)",
        },
    },
    "roberta": {
        "accuracy": 0.9489,
        "precision": 0.5645,
        "recall": 0.5692,
        "f1": 0.5590,
        "parameters": "125M",
        "architecture": "Transformer",
        "training_time": "38.8 мин",
        "training_config": {
            "epochs": 5,
            "learning_rate": "2e-5",
            "batch_size": 32,
            "max_length": 128,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "dropout": 0.1,
            "dataset": "UD English EWT + GUM (~489k tokens, 406 labels)",
        },
    },
    "albert": {
        "accuracy": 0.9464,
        "precision": 0.5771,
        "recall": 0.5611,
        "f1": 0.5609,
        "parameters": "12M",
        "architecture": "Transformer",
        "training_time": "49.0 мин",
        "training_config": {
            "epochs": 6,
            "learning_rate": "2e-5",
            "batch_size": 64,
            "max_length": 128,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "dropout": 0.1,
            "dataset": "UD English EWT + GUM (~489k tokens, 406 labels)",
        },
    },
    "flair": {
        "accuracy": 0.9395,
        "precision": 0.5492,
        "recall": 0.5121,
        "f1": 0.5208,
        "parameters": "63M",
        "architecture": "BiLSTM+CRF",
        "training_time": "~30 мин",
        "training_config": {
            "epochs": 5,
            "learning_rate": "0.1",
            "batch_size": 32,
            "max_length": None,
            "warmup_ratio": None,
            "weight_decay": None,
            "dropout": None,
            "dataset": "UD English EWT + GUM (~489k tokens, 406 labels)",
        },
    },
}


@router.get("/models", response_model=ModelsResponse)
async def list_models():
    """
    Get list of available morphological analysis models.
    
    Returns information about each model including:
    - id: Model identifier for use in /analyze endpoint
    - name: Display name
    - description: Brief description
    - loaded: Whether the model is currently loaded in memory
    - Evaluation metrics (accuracy, precision, recall, f1, etc.)
    """
    models = model_service.get_available_models()
    
    enriched_models = []
    for m in models:
        metrics = MODEL_METRICS.get(m["id"], {})
        enriched_models.append(ModelInfo(**m, **metrics))
    
    return ModelsResponse(
        models=enriched_models,
        default_model=settings.default_model
    )
