"""
Models API routes - list available models
"""
from fastapi import APIRouter

from app.core.config import settings
from app.services.model_service import model_service
from app.schemas.analysis import ModelsResponse, ModelInfo

router = APIRouter()


@router.get("/models", response_model=ModelsResponse)
async def list_models():
    """
    Get list of available morphological analysis models.
    
    Returns information about each model including:
    - id: Model identifier for use in /analyze endpoint
    - name: Display name
    - description: Brief description
    - loaded: Whether the model is currently loaded in memory
    """
    models = model_service.get_available_models()
    
    return ModelsResponse(
        models=[ModelInfo(**m) for m in models],
        default_model=settings.default_model
    )
