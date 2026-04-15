"""
H1 with Pre-trained Architecture (Simplified)

Simplified version that uses random embeddings as placeholder for pre-trained models.
This allows testing the architecture without network dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class SimplePretrainedEncoder(nn.Module):
    """Simple encoder that mimics pre-trained behavior with random embeddings."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, embedding_dim: int = 768):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Initialize with Xavier uniform (similar to pre-trained models)
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input."""
        encoded = self.encoder(x)
        return F.normalize(encoded, dim=1)

class H1SimplifiedPretrained(nn.Module):
    """
    H1 with Simplified Pre-trained Architecture
    
    Uses random embeddings as placeholder for pre-trained models.
    Architecture is designed to work with pre-trained embeddings when available.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Patient encoder (mimics pre-trained behavior)
        self.patient_encoder = SimplePretrainedEncoder(
            input_dim=config.confounder_dim + config.treatment_dim,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim
        )
        
        # Treatment encoder
        self.treatment_encoder = nn.Linear(config.treatment_dim, config.hidden_dim)
        
        # Confounder encoder
        self.confounder_encoder = nn.Linear(config.confounder_dim, config.hidden_dim)
        
        # Retrieval encoder
        self.retrieval_encoder = nn.Linear(
            config.embedding_dim * config.retrieval_top_k,
            config.hidden_dim
        )
        
        # Outcome predictor
        self.outcome_predictor = nn.Sequential(
            nn.Linear(3 * config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.outcome_dim)
        )
        
        # Attribution network
        self.attribution_network = nn.Sequential(
            nn.Linear(config.hidden_dim + config.outcome_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.retrieval_top_k),
            nn.Softmax(dim=1)
        )
    
    def forward(self, batch: Dict, corpus_embeddings: torch.Tensor = None) -> Dict:
        """Forward pass."""
        patient_features = batch['patient']
        treatment = batch['treatment']
        confounders = batch['confounders']
        
        batch_size = patient_features.size(0)
        
        # Encode patient
        patient_emb = self.patient_encoder(patient_features)
        
        # Retrieve using patient embedding
        if corpus_embeddings is not None:
            # Normalize embeddings
            query_emb = F.normalize(patient_emb, dim=1)
            corpus_emb = F.normalize(corpus_embeddings, dim=1)
            
            # Compute similarity
            similarity = torch.matmul(query_emb, corpus_emb.t())
            
            # Get top-k documents
            top_k = min(self.config.retrieval_top_k, similarity.size(1))
            retrieval_scores, retrieval_indices = torch.topk(similarity, k=top_k, dim=1)
            
            # Gather retrieved embeddings
            expanded_indices = retrieval_indices.unsqueeze(-1).expand(-1, -1, self.config.embedding_dim)
            retrieved_embeddings = torch.gather(
                corpus_embeddings.unsqueeze(0).expand(batch_size, -1, -1),
                1,
                expanded_indices
            )
        else:
            # Fallback: random retrieval
            retrieved_embeddings = torch.randn(
                batch_size, self.config.retrieval_top_k, self.config.embedding_dim,
                device=patient_features.device
            )
            retrieval_scores = torch.zeros(batch_size, self.config.retrieval_top_k, device=patient_features.device)
            retrieval_indices = torch.zeros(batch_size, self.config.retrieval_top_k, dtype=torch.long, device=patient_features.device)
        
        # Flatten retrieval
        flattened_retrieval = retrieved_embeddings.view(batch_size, -1)
        
        # Encode components
        treatment_enc = self.treatment_encoder(treatment)
        confounder_enc = self.confounder_encoder(confounders)
        retrieval_enc = self.retrieval_encoder(flattened_retrieval)
        
        # Combine for prediction
        combined = torch.cat([treatment_enc, confounder_enc, retrieval_enc], dim=1)
        outcome = self.outcome_predictor(combined)
        
        # Compute attribution scores
        attribution_input = torch.cat([retrieval_enc, outcome], dim=1)
        attribution_scores = self.attribution_network(attribution_input)
        
        return {
            'outcome': outcome,
            'retrieval_scores': retrieval_scores,
            'retrieval_indices': retrieval_indices,
            'attribution_scores': attribution_scores,
            'patient_embedding': patient_emb
        }
    
    def compute_loss(self, pred: Dict, batch: Dict) -> torch.Tensor:
        """Compute loss."""
        # Prediction loss
        prediction_loss = F.mse_loss(pred['outcome'], batch['outcome'])
        
        # Attribution consistency loss
        if 'retrieval_scores' in pred and 'attribution_scores' in pred:
            retrieval_scores = pred['retrieval_scores']
            attribution_scores = pred['attribution_scores']
            alignment_loss = F.mse_loss(attribution_scores, retrieval_scores)
        else:
            alignment_loss = torch.tensor(0.0, device=prediction_loss.device)
        
        total_loss = prediction_loss + 0.1 * alignment_loss
        
        return total_loss

def create_simple_corpus_embeddings(n_docs: int = 500, embedding_dim: int = 768):
    """Create simple random corpus embeddings."""
    print(f"Creating simple corpus embeddings ({n_docs} docs, {embedding_dim} dim)...")
    
    embeddings = np.random.randn(n_docs, embedding_dim).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    
    return torch.FloatTensor(embeddings)

# Example usage
if __name__ == "__main__":
    from config import Config
    
    # Create config
    config = Config()
    config.confounder_dim = 50
    config.treatment_dim = 10
    config.outcome_dim = 1
    config.embedding_dim = 768
    config.hidden_dim = 256
    config.retrieval_top_k = 5
    
    # Create model
    print("Creating H1 with simplified pre-trained architecture...")
    model = H1SimplifiedPretrained(config)
    
    # Test forward pass
    batch = {
        'patient': torch.randn(4, config.confounder_dim + config.treatment_dim),
        'treatment': torch.randn(4, config.treatment_dim),
        'confounders': torch.randn(4, config.confounder_dim),
        'outcome': torch.randn(4, config.outcome_dim)
    }
    
    # Create corpus embeddings
    corpus_embeddings = create_simple_corpus_embeddings(n_docs=100, embedding_dim=768)
    
    # Forward pass
    pred = model(batch, corpus_embeddings)
    
    print("\nH1 with Simplified Pre-trained Architecture Test:")
    print(f"  Outcome shape: {pred['outcome'].shape}")
    print(f"  Patient embedding shape: {pred['patient_embedding'].shape}")
    print(f"  Retrieval scores shape: {pred['retrieval_scores'].shape}")
    
    # Compute loss
    loss = model.compute_loss(pred, batch)
    print(f"  Loss: {loss.item():.4f}")
    
    print("\n✓ H1 with simplified pre-trained architecture initialized successfully!")
