"""
Simplified PICO-Contrastive RAG

Simplified implementation that works with existing data format.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class SimplePICOContrastiveRAG(nn.Module):
    """
    Simplified PICO-Contrastive RAG.
    
    Uses contrastive embeddings for treatment effect similarity.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Patient encoder (population)
        self.patient_encoder = nn.Sequential(
            nn.Linear(config.confounder_dim + config.treatment_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Treatment encoder (intervention)
        self.treatment_encoder = nn.Linear(config.treatment_dim, config.hidden_dim)
        
        # Confounder encoder (baseline)
        self.confounder_encoder = nn.Linear(config.confounder_dim, config.hidden_dim)
        
        # Contrastive embedding generator
        self.contrastive_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )
        
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
        
        # Contrastive margin
        self.margin = 0.5
    
    def forward(self, batch: Dict, corpus_embeddings: torch.Tensor = None) -> Dict:
        """Forward pass with contrastive retrieval."""
        patient_features = batch['patient']
        treatment = batch['treatment']
        confounders = batch['confounders']
        patient_query_emb = batch.get('patient_query_emb', None)
        
        batch_size = patient_features.size(0)
        
        # Encode patient (population)
        patient_emb = self.patient_encoder(patient_features)
        
        # Encode treatment (intervention)
        treatment_emb = self.treatment_encoder(treatment)
        
        # Encode confounders (baseline)
        confounder_emb = self.confounder_encoder(confounders)
        
        # Generate contrastive embedding: f(P,I) - f(P,C)
        # For simplicity, use patient_emb - confounder_emb as contrastive
        contrastive_emb = self.contrastive_encoder(
            torch.cat([patient_emb, confounder_emb], dim=1)
        )
        
        # Retrieve using contrastive embedding
        if corpus_embeddings is not None:
            # Normalize embeddings
            query_emb = F.normalize(contrastive_emb, dim=1)
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
        retrieval_enc = self.retrieval_encoder(flattened_retrieval)
        
        # Combine for prediction
        combined = torch.cat([treatment_emb, confounder_emb, retrieval_enc], dim=1)
        outcome = self.outcome_predictor(combined)
        
        return {
            'outcome': outcome,
            'retrieval_scores': retrieval_scores,
            'retrieval_indices': retrieval_indices,
            'contrastive_embedding': contrastive_emb
        }
    
    def compute_loss(self, pred: Dict, batch: Dict) -> torch.Tensor:
        """Compute loss with contrastive regularization."""
        # Prediction loss
        prediction_loss = F.mse_loss(pred['outcome'], batch['outcome'])
        
        # Contrastive loss (simplified)
        if 'contrastive_embedding' in pred:
            # Use retrieval scores as proxy for contrastive loss
            retrieval_scores = pred['retrieval_scores']
            contrastive_loss = -torch.log(retrieval_scores + 1e-8).mean()
        else:
            contrastive_loss = torch.tensor(0.0, device=prediction_loss.device)
        
        # Total loss
        total_loss = prediction_loss + 0.3 * contrastive_loss
        
        return total_loss

def create_simple_pico_contrastive_rag(config):
    """Factory function to create simplified PICO-Contrastive RAG."""
    return SimplePICOContrastiveRAG(config)

# Example usage
if __name__ == "__main__":
    from config import Config
    
    # Create config
    config = Config()
    config.confounder_dim = 50
    config.treatment_dim = 8
    config.outcome_dim = 1
    config.embedding_dim = 768
    config.hidden_dim = 256
    config.retrieval_top_k = 5
    
    # Create model
    model = create_simple_pico_contrastive_rag(config)
    
    # Test forward pass
    batch = {
        'patient': torch.randn(4, config.confounder_dim + config.treatment_dim),
        'treatment': torch.randn(4, config.treatment_dim),
        'confounders': torch.randn(4, config.confounder_dim),
        'outcome': torch.randn(4, config.outcome_dim)
    }
    
    corpus_embeddings = torch.randn(100, config.embedding_dim)
    
    # Forward pass
    pred = model(batch, corpus_embeddings)
    
    print("Simplified PICO-Contrastive RAG Test:")
    print(f"  Outcome shape: {pred['outcome'].shape}")
    print(f"  Contrastive embedding shape: {pred['contrastive_embedding'].shape}")
    print(f"  Retrieval scores shape: {pred['retrieval_scores'].shape}")
    
    # Compute loss
    loss = model.compute_loss(pred, batch)
    print(f"  Loss: {loss.item():.4f}")
