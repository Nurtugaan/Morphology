"""
Configuration settings for the Morphology Analyzer API
"""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    app_name: str = "Morphology Analyzer API"
    app_version: str = "1.0.0"
    debug: bool = True
    
    # Model paths (relative to project root)
    models_base_path: Path = Path(__file__).parent.parent.parent.parent / "services" / "models"
    
    # Available models configuration
    available_models: dict = {
        "distilbert": {
            "name": "DistilBERT",
            "path": "en/distilbert_model/distilbert_model/best_model",
            "base_model": "distilbert-base-uncased",
            "type": "transformer",
            "description": "Fast & lightweight (66M params)"
        },
        "bert": {
            "name": "BERT-base",
            "path": "en/bert-base_model/best_model",
            "base_model": "bert-base-uncased",
            "type": "transformer",
            "description": "Standard accuracy (110M params)"
        },
        "roberta": {
            "name": "RoBERTa-base",
            "path": "en/roberta-base_model/best_model",
            "base_model": "roberta-base",
            "type": "transformer",
            "description": "Improved BERT (125M params)"
        },
        "albert": {
            "name": "ALBERT-base",
            "path": "en/albert-base_model/best_model",
            "base_model": "albert-base-v2",
            "type": "transformer",
            "description": "Parameter efficient (12M params)"
        },
        "flair": {
            "name": "Flair",
            "path": "en/flair_model",
            "base_model": None,
            "type": "flair",
            "description": "BiLSTM + CRF + Flair Embeddings"
        }
    }
    
    # Inference settings
    max_text_length: int = 512
    default_model: str = "distilbert"
    
    class Config:
        env_prefix = "MORPHO_"


settings = Settings()
