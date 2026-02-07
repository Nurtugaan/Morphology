"""
Morphology Analyzer API

FastAPI application for interactive testing of morphological analysis models.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes import models, analysis

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="""
## Morphology Analyzer API

Интерактивный сервис для морфологического анализа текста.

### Возможности:
- Выбор модели из доступных (DistilBERT, BERT, RoBERTa, ALBERT)
- Морфологический анализ текста с определением:
  - Части речи (POS tags)
  - Морфологических признаков (Number, Tense, Case, etc.)

### Примеры использования:
1. Получить список моделей: `GET /api/models`
2. Анализ текста: `POST /api/analyze` с JSON `{"text": "...", "model_name": "distilbert"}`
    """,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(models.router, prefix="/api", tags=["Models"])
app.include_router(analysis.router, prefix="/api", tags=["Analysis"])


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    from app.services.model_service import model_service
    
    return {
        "status": "healthy",
        "device": model_service.device,
        "loaded_models": list(model_service.models.keys()),
        "available_models": list(settings.available_models.keys())
    }
