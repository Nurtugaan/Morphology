"""
Pydantic schemas for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import Optional


class AnalyzeRequest(BaseModel):
    """Request schema for text analysis"""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")
    model_name: str = Field(default="distilbert", description="Model to use: distilbert, bert, roberta, albert")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "The cat sat on the mat.",
                "model_name": "distilbert"
            }
        }


class TokenAnalysis(BaseModel):
    """Analysis result for a single token"""
    token: str = Field(..., description="Original word/token")
    upos: str = Field(..., description="Universal POS tag (NOUN, VERB, etc.)")
    features: dict = Field(default_factory=dict, description="Morphological features")
    label: str = Field(..., description="Full label in UPOS|Feature=Value format")


class AnalyzeResponse(BaseModel):
    """Response schema for text analysis"""
    model_name: str = Field(..., description="Model used for analysis")
    text: str = Field(..., description="Original input text")
    tokens: list[TokenAnalysis] = Field(..., description="Analysis results per token")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "distilbert",
                "text": "The cat sat.",
                "tokens": [
                    {"token": "The", "upos": "DET", "features": {"Definite": "Def"}, "label": "DET|Definite=Def"},
                    {"token": "cat", "upos": "NOUN", "features": {"Number": "Sing"}, "label": "NOUN|Number=Sing"},
                    {"token": "sat", "upos": "VERB", "features": {"Tense": "Past"}, "label": "VERB|Tense=Past"},
                    {"token": ".", "upos": "PUNCT", "features": {}, "label": "PUNCT|_"}
                ],
                "processing_time_ms": 45.2
            }
        }


class ModelInfo(BaseModel):
    """Information about an available model"""
    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Display name")
    description: str = Field(..., description="Model description")
    loaded: bool = Field(default=False, description="Whether model is currently loaded")


class ModelsResponse(BaseModel):
    """Response schema for available models list"""
    models: list[ModelInfo] = Field(..., description="List of available models")
    default_model: str = Field(..., description="Default model ID")


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
