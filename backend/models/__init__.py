"""
models/__init__.py — Model Manager Abstraction Layer
====================================================
Centralized lazy-loading, caching, and lifecycle management for all ML models.
Eliminates duplicate boilerplate across modules.
"""

import logging
import threading
from typing import Optional, Tuple, Any, Callable
from functools import wraps

log = logging.getLogger(__name__)


class ModelManager:
    """
    Thread-safe model registry with lazy loading and optional preloading.
    
    Usage:
        manager = ModelManager()
        
        # Register model loaders
        manager.register("sbert", lambda: SentenceTransformer("BAAI/bge-small-en-v1.5"))
        manager.register("pegasus", lambda: (PegasusTokenizer.from_pretrained(...), PegasusForConditionalGeneration.from_pretrained(...)))
        
        # Get models (lazy-loaded on first access)
        sbert = manager.get("sbert")
        tokenizer, model = manager.get("pegasus")
        
        # Preload at startup (optional)
        manager.preload(["sbert", "pegasus"])
    """
    
    def __init__(self):
        self._models: dict[str, Any] = {}
        self._loaders: dict[str, Callable[[], Any]] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._load_order: list[str] = []  # Track registration order for preloading
    
    def register(self, name: str, loader: Callable[[], Any]) -> "ModelManager":
        """
        Register a model loader function.
        
        Args:
            name: Unique identifier for the model
            loader: Callable that returns the loaded model (or tuple of models)
            
        Returns:
            Self for chaining
        """
        with self._global_lock:
            if name in self._loaders:
                log.warning(f"Model '{name}' already registered, overwriting")
            self._loaders[name] = loader
            self._locks[name] = threading.Lock()
            if name not in self._load_order:
                self._load_order.append(name)
        return self
    
    def get(self, name: str) -> Any:
        """
        Get a model, loading it lazily on first access.
        Thread-safe: only one thread will execute the loader.
        
        Args:
            name: Model identifier
            
        Returns:
            The loaded model (or tuple)
            
        Raises:
            KeyError: If model not registered
            Exception: Any exception from the loader
        """
        # Fast path: already loaded
        if name in self._models:
            return self._models[name]
        
        # Get or create lock for this model
        with self._global_lock:
            if name not in self._locks:
                raise KeyError(f"Model '{name}' not registered. Available: {list(self._loaders.keys())}")
            lock = self._locks[name]
        
        # Double-checked locking
        with lock:
            if name in self._models:
                return self._models[name]
            
            if name not in self._loaders:
                raise KeyError(f"Model '{name}' not registered")
            
            log.info(f"Loading model '{name}'...")
            try:
                model = self._loaders[name]()
                self._models[name] = model
                log.info(f"Model '{name}' loaded successfully")
                return model
            except Exception as e:
                log.error(f"Failed to load model '{name}': {e}")
                raise
    
    def preload(self, names: Optional[list[str]] = None) -> dict[str, bool]:
        """
        Preload models (useful at startup to avoid first-request latency).
        
        Args:
            names: List of model names to preload. If None, preload all registered.
            
        Returns:
            Dict mapping model name to success status
        """
        if names is None:
            names = self._load_order.copy()
        
        results = {}
        for name in names:
            try:
                self.get(name)
                results[name] = True
            except Exception as e:
                log.error(f"Preload failed for '{name}': {e}")
                results[name] = False
        return results
    
    def is_loaded(self, name: str) -> bool:
        """Check if a model is already loaded in memory."""
        return name in self._models
    
    def unload(self, name: str) -> bool:
        """
        Unload a model from memory (frees GPU/CPU memory).
        
        Returns:
            True if model was loaded and unloaded, False if not loaded
        """
        with self._global_lock:
            if name in self._models:
                # Try to free memory for torch models
                model = self._models[name]
                if hasattr(model, 'cpu'):
                    try:
                        model.cpu()
                    except Exception:
                        pass
                del self._models[name]
                log.info(f"Model '{name}' unloaded")
                return True
        return False
    
    def unload_all(self) -> int:
        """Unload all models. Returns count of unloaded models."""
        count = 0
        for name in list(self._models.keys()):
            if self.unload(name):
                count += 1
        return count
    
    def list_registered(self) -> list[str]:
        """Get list of registered model names."""
        return self._load_order.copy()
    
    def list_loaded(self) -> list[str]:
        """Get list of currently loaded model names."""
        return list(self._models.keys())


# Global instance for convenience
_default_manager: Optional[ModelManager] = None
_default_manager_lock = threading.Lock()


def get_model_manager() -> ModelManager:
    """Get the global model manager instance (singleton)."""
    global _default_manager
    if _default_manager is None:
        with _default_manager_lock:
            if _default_manager is None:
                _default_manager = ModelManager()
    return _default_manager


def register_default_models() -> ModelManager:
    """
    Register all default ScholarAI models.
    Call this at startup to configure the global manager.
    """
    manager = get_model_manager()
    
    # SBERT (BGE-small) - shared between summarizer and quiz generator
    manager.register("sbert", lambda: _load_sbert())
    
    # Pegasus summarization
    manager.register("pegasus", lambda: _load_pegasus())
    
    # T5 question generation
    manager.register("t5", lambda: _load_t5())
    
    # Autoencoder
    manager.register("autoencoder", lambda: _load_autoencoder())
    
    return manager


# ─── PRIVATE LOADER FUNCTIONS ────────────────────────────────────

def _load_sbert():
    """Load BGE sentence transformer (base or small based on config)."""
    from sentence_transformers import SentenceTransformer
    from ..config import get_settings
    settings = get_settings()
    
    if settings.MODEL_TIER == "speed":
        model_name = settings.SBERT_MODEL_LIGHT
    else:
        model_name = settings.SBERT_MODEL
    
    log.info(f"Loading SBERT model: {model_name}…")
    return SentenceTransformer(model_name)


def _load_pegasus():
    """Load BART-large-CNN tokenizer and model for summarization."""
    from transformers import BartTokenizer, BartForConditionalGeneration
    from ..config import get_settings
    settings = get_settings()
    
    if settings.MODEL_TIER == "speed":
        model_name = settings.PEGASUS_MODEL_LIGHT
    else:
        model_name = settings.PEGASUS_MODEL
    
    log.info(f"Loading summarization model: {model_name}…")
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer


def _load_t5():
    """Load FLAN-T5 tokenizer and model for question generation."""
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    from ..config import get_settings
    settings = get_settings()
    
    if settings.MODEL_TIER == "speed":
        model_name = settings.T5_MODEL_LIGHT
    else:
        model_name = settings.T5_MODEL
    
    log.info(f"Loading QG model: {model_name}…")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    log.info("FLAN-T5 QG model loaded.")
    return model, tokenizer


def _load_autoencoder(input_dim: int = 768):
    """Load semantic autoencoder."""
    from .autoencoder import SemanticAutoencoder
    from ..config import get_settings
    settings = get_settings()
    latent_dim = settings.AUTOENCODER_LATENT_DIM
    ae = SemanticAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    ae.try_load_weights()
    return ae


# ─── CONVENIENCE FUNCTIONS ───────────────────────────────────────

def get_sbert():
    """Get SBERT model (BGE-small)."""
    return get_model_manager().get("sbert")


def get_pegasus() -> Tuple[Any, Any]:
    """Get Pegasus model and tokenizer."""
    return get_model_manager().get("pegasus")


def get_t5() -> Tuple[Any, Any]:
    """Get T5 model and tokenizer."""
    return get_model_manager().get("t5")


def get_autoencoder(input_dim: int = 384):
    """Get autoencoder instance."""
    # Autoencoder needs input_dim, so we handle it specially
    manager = get_model_manager()
    if not manager.is_loaded("autoencoder"):
        # Register with specific input_dim if not already loaded
        manager.register("autoencoder", lambda: _load_autoencoder(input_dim))
    return manager.get("autoencoder")


# ─── DECORATOR FOR AUTO-PRELOADING ───────────────────────────────

def with_models(*model_names: str):
    """
    Decorator that ensures models are loaded before function execution.
    Useful for warming up models in background threads.
    
    Usage:
        @with_models("sbert", "pegasus")
        def generate_summary(text):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_model_manager()
            for name in model_names:
                if not manager.is_loaded(name):
                    manager.get(name)
            return func(*args, **kwargs)
        return wrapper
    return decorator