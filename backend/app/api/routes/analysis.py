"""
Analysis API routes - morphological analysis endpoint
"""
from fastapi import APIRouter, HTTPException

from app.services.model_service import model_service
from app.schemas.analysis import AnalyzeRequest, AnalyzeResponse, ErrorResponse

router = APIRouter()


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Model not found"},
        500: {"model": ErrorResponse, "description": "Inference error"}
    }
)
async def analyze_text(request: AnalyzeRequest):
    """
    Perform morphological analysis on text.
    
    Analyzes the input text and returns morphological information for each token:
    - **token**: Original word
    - **upos**: Universal POS tag (NOUN, VERB, ADJ, etc.)
    - **features**: Morphological features (Number, Tense, Case, etc.)
    - **label**: Full label in UPOS|Feature=Value format
    
    Available models:
    - `distilbert`: Fast & lightweight (recommended)
    - `bert`: Standard BERT-base
    - `roberta`: RoBERTa-base
    - `albert`: ALBERT-base (parameter efficient)
    """
    try:
        result = model_service.analyze(
            text=request.text,
            model_name=request.model_name
        )
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
