"""
Hyperparameter configuration for causal RAG experiments.
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Config:
    """Configuration for causal RAG experiments."""
    
    # Training hyperparameters
    lr: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    
    # Model architecture
    hidden_dim: int = 256
    retrieval_top_k: int = 5
    embedding_dim: int = 768
    max_seq_len: int = 512
    
    # Causal inference dimensions
    treatment_dim: int = 10
    outcome_dim: int = 1
    confounder_dim: int = 50
    
    # Experiment settings
    num_seeds: int = 3
    max_gpu_hours: float = 100.0
    
    # Causal regularization
    causal_lambda: float = 0.1
    iv_strength_threshold: float = 0.3
    
    # Dataset paths (for compatibility with data loading code)
    data_root: str = "/workspace/data"
    
    # Device configuration
    device: str = "mps"  # Apple M4 Pro GPU
    
    # Model selection
    model_type: str = "CausalRAGWithAdjustment"  # Default model
    
    # Evaluation metrics
    primary_metric_weights: tuple = (0.6, 0.3, 0.1)  # (1-ATE_error, AUC_ROC, retrieval_precision)
    
    # Additional parameters for compatibility
    SEED: int = 42
    N_PATIENTS: int = 2000
    N_FEATURES: int = 10
    N_TREATMENTS: int = 3
    N_RETRIEVAL_DOCS: int = 100
    DOC_EMBEDDING_DIM: int = 128
    PROPENSITY_SCORE_METHOD: str = 'logistic'
    CAUSAL_ESTIMATOR: str = 'doubly_robust'
    RETRIEVAL_TOP_K: int = 5
    RERANK_TOP_K: int = 3
    HIDDEN_DIM: int = 64
    N_EPOCHS: int = 15
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    TEST_SIZE: float = 0.2
    N_BOOTSTRAP: int = 100
    
    # Architecture-specific hyperparameters
    # CFR CausalRAG
    CFR_BALANCE_LAMBDA: float = 1.0
    CFR_BALANCE_METHOD: str = 'mmd'
    
    # Enhanced CausalRAG - OPTIMIZED for causal effect learning
    # cf_lambda controls counterfactual prediction loss weight
    # Higher values (5-10) force model to learn treatment effects
    ENHANCED_CF_LAMBDA: float = 5.0
    ENHANCED_ATE_LAMBDA: float = 1.0
    ENHANCED_BALANCE_LAMBDA: float = 0.1
    ENHANCED_RETRIEVAL_LAMBDA: float = 0.2
    
    # Dragonnet CausalRAG
    DRAGONNET_ALPHA: float = 1.0
    
    # MultiHead CausalRAG - OPTIMIZED
    MULTIHEAD_CF_LAMBDA: float = 5.0
    MULTIHEAD_ATE_LAMBDA: float = 1.0
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)
    
    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# Global hyperparameters dictionary for reporting
HYPERPARAMETERS = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 50,
    'hidden_dim': 256,
    'retrieval_top_k': 5,
    'embedding_dim': 768,
    'max_seq_len': 512,
    'treatment_dim': 10,
    'outcome_dim': 1,
    'confounder_dim': 50,
    'num_seeds': 3,
    'max_gpu_hours': 100.0,
    'causal_lambda': 0.1,
    'iv_strength_threshold': 0.3,
    'data_root': "/workspace/data",
    'device': "mps",
    'model_type': "CausalRAGWithAdjustment",
    'primary_metric_weights': (0.6, 0.3, 0.1)
}


def get_config(**kwargs) -> Config:
    """
    Create a Config instance with optional overrides.
    
    Args:
        **kwargs: Configuration parameters to override
        
    Returns:
        Config instance
    """
    config = Config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Invalid config parameter: {key}")
    return config


# Default configuration
default_config = Config()