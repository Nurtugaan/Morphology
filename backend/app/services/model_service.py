"""
Model Service for loading and running inference on morphological models

Поддерживает:
- Transformer модели (BERT, RoBERTa, DistilBERT, ALBERT)
- Flair модели (BiLSTM + CRF + Flair Embeddings)
"""
import json
import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Union
from transformers import AutoTokenizer, AutoModel, AutoConfig
import sys

# Add services to path for model imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "services"))

from app.core.config import settings
from app.schemas.analysis import TokenAnalysis, AnalyzeResponse


class MorphologyTaggerInference(nn.Module):
    """
    Модель для инференса морфологического анализа (Transformer).
    
    Загружает encoder из локальной папки (если есть model.safetensors) 
    или из HuggingFace + classifier из classifier.pt
    """
    
    def __init__(self, encoder, num_labels: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def load_classifier(self, classifier_path: Path):
        """Load classifier weights from file"""
        state = torch.load(classifier_path, map_location='cpu', weights_only=True)
        self.classifier.load_state_dict(state['classifier'])
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits
    
    def predict(self, input_ids, attention_mask):
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            predictions = logits.argmax(dim=-1)
        return predictions


class ModelService:
    """
    Service for managing morphological analysis models.
    """
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.label_mappings: Dict[str, Dict[int, str]] = {}
        self.model_types: Dict[str, str] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ModelService] Initialized with device: {self.device}")
    
    def get_available_models(self) -> list:
        """Returns list of available model configurations"""
        return [
            {
                "id": model_id,
                "name": config["name"],
                "description": config["description"],
                "loaded": model_id in self.models
            }
            for model_id, config in settings.available_models.items()
        ]
    
    def is_model_loaded(self, model_name: str) -> bool:
        return model_name in self.models
    
    def load_transformer_model(self, model_name: str, config: dict, model_path: Path):
        """Load a transformer-based model"""
        print(f"[ModelService] Loading transformer model '{model_name}'...", flush=True)
        print(f"  Model path: {model_path}", flush=True)
        
        # Load id2label mapping
        id2label_path = model_path / "id2label.json"
        with open(id2label_path, "r", encoding="utf-8") as f:
            id2label = {int(k): v for k, v in json.load(f).items()}
        
        # Load classifier state to get num_labels
        classifier_path = model_path / "classifier.pt"
        classifier_state = torch.load(classifier_path, map_location='cpu', weights_only=True)
        num_labels = classifier_state['num_labels']
        hidden_size = classifier_state['hidden_size']
        dropout = classifier_state.get('dropout_prob', 0.1)
        
        base_model = config["base_model"]
        
        # Check if fine-tuned encoder exists locally (model.safetensors or pytorch_model.bin)
        safetensors_path = model_path / "model.safetensors"
        pytorch_path = model_path / "pytorch_model.bin"
        
        print(f"  Checking for local encoder:", flush=True)
        print(f"    model.safetensors exists: {safetensors_path.exists()}", flush=True)
        print(f"    pytorch_model.bin exists: {pytorch_path.exists()}", flush=True)
        
        has_local_encoder = safetensors_path.exists() or pytorch_path.exists()
        
        if has_local_encoder:
            print(f"  ✓ Loading FINE-TUNED encoder from: {model_path}", flush=True)
            encoder = AutoModel.from_pretrained(str(model_path))
        else:
            print(f"  ⚠ WARNING: No fine-tuned encoder found!", flush=True)
            print(f"  Loading BASE model from HuggingFace: {base_model}", flush=True)
            print(f"  Results will likely be poor!", flush=True)
            encoder = AutoModel.from_pretrained(base_model)
        
        # Load tokenizer
        if "roberta" in base_model.lower():
            tokenizer = AutoTokenizer.from_pretrained(base_model, add_prefix_space=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Create model and load classifier
        model = MorphologyTaggerInference(
            encoder=encoder,
            num_labels=num_labels,
            hidden_size=hidden_size,
            dropout=dropout
        )
        model.load_classifier(classifier_path)
        model.to(self.device)
        model.eval()
        
        return model, tokenizer, id2label
    
    def load_flair_model(self, model_name: str, config: dict, model_path: Path):
        """Load a Flair SequenceTagger model"""
        print(f"[ModelService] Loading Flair model '{model_name}'...")
        
        try:
            from flair.models import SequenceTagger
        except ImportError:
            raise ImportError("Flair library not installed. Run: pip install flair")
        
        flair_model_file = model_path / "best-model.pt"
        if not flair_model_file.exists():
            flair_model_file = model_path / "final-model.pt"
        
        if not flair_model_file.exists():
            raise FileNotFoundError(f"Flair model not found at: {model_path}")
        
        model = SequenceTagger.load(flair_model_file)
        return model, None, {}
    
    def load_model(self, model_name: str):
        """Load a model if not already in cache."""
        if model_name not in settings.available_models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(settings.available_models.keys())}")
        
        if model_name in self.models:
            return self.models[model_name], self.tokenizers.get(model_name), self.label_mappings.get(model_name, {})
        
        config = settings.available_models[model_name]
        model_path = settings.models_base_path / config["path"]
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        start_time = time.time()
        model_type = config.get("type", "transformer")
        
        if model_type == "flair":
            model, tokenizer, id2label = self.load_flair_model(model_name, config, model_path)
        else:
            model, tokenizer, id2label = self.load_transformer_model(model_name, config, model_path)
        
        # Cache
        self.models[model_name] = model
        self.tokenizers[model_name] = tokenizer
        self.label_mappings[model_name] = id2label
        self.model_types[model_name] = model_type
        
        load_time = time.time() - start_time
        print(f"[ModelService] Model '{model_name}' loaded in {load_time:.2f}s")
        
        return model, tokenizer, id2label
    
    def parse_label(self, label: str) -> Tuple[str, dict]:
        """Parse a label in format 'UPOS|Feature=Value' into parts."""
        if "|" not in label:
            return label, {}
        
        parts = label.split("|")
        upos = parts[0]
        
        features = {}
        for part in parts[1:]:
            if "=" in part:
                key, value = part.split("=", 1)
                features[key] = value
            elif part != "_":
                features[part] = True
        
        return upos, features
    
    def analyze_with_transformer(self, text: str, model_name: str, model, tokenizer, id2label) -> AnalyzeResponse:
        """Analyze text using a transformer model"""
        start_time = time.time()
        
        words = text.split()
        
        encoding = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=settings.max_text_length,
            padding=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        with torch.no_grad():
            predictions = model.predict(input_ids, attention_mask)
        
        word_ids = encoding.word_ids()
        
        token_analyses = []
        seen_word_ids = set()
        
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id in seen_word_ids:
                continue
            seen_word_ids.add(word_id)
            
            pred_id = predictions[0, idx].item()
            label = id2label.get(pred_id, "O")
            upos, features = self.parse_label(label)
            
            token_analyses.append(TokenAnalysis(
                token=words[word_id],
                upos=upos,
                features=features,
                label=label
            ))
        
        processing_time = (time.time() - start_time) * 1000
        
        return AnalyzeResponse(
            model_name=model_name,
            text=text,
            tokens=token_analyses,
            processing_time_ms=round(processing_time, 2)
        )
    
    def analyze_with_flair(self, text: str, model_name: str, model) -> AnalyzeResponse:
        """Analyze text using a Flair model"""
        from flair.data import Sentence
        
        start_time = time.time()
        
        sentence = Sentence(text)
        model.predict(sentence)
        
        token_analyses = []
        for token in sentence:
            label = token.get_label('pos').value if token.get_label('pos') else "O"
            upos, features = self.parse_label(label)
            
            token_analyses.append(TokenAnalysis(
                token=token.text,
                upos=upos,
                features=features,
                label=label
            ))
        
        processing_time = (time.time() - start_time) * 1000
        
        return AnalyzeResponse(
            model_name=model_name,
            text=text,
            tokens=token_analyses,
            processing_time_ms=round(processing_time, 2)
        )
    
    def analyze(self, text: str, model_name: str = "distilbert") -> AnalyzeResponse:
        """Perform morphological analysis on text."""
        model, tokenizer, id2label = self.load_model(model_name)
        model_type = self.model_types.get(model_name, "transformer")
        
        if model_type == "flair":
            return self.analyze_with_flair(text, model_name, model)
        else:
            return self.analyze_with_transformer(text, model_name, model, tokenizer, id2label)


# Singleton instance
model_service = ModelService()
