"""
PICO-Contrastive RAG

Implements PICO-structured retrieval with contrastive embeddings for treatment effect similarity.

PICO Structure:
- Population: Patient characteristics
- Intervention: Treatment being studied
- Comparison: Control/comparator treatment
- Outcome: Measured outcome

Contrastive Embedding: f(P,I) - f(P,C) for outcome O
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class PICOStructure:
    """PICO structure for clinical queries."""
    population: torch.Tensor  # Patient features
    intervention: torch.Tensor  # Treatment features
    comparison: torch.Tensor  # Control/comparator features
    outcome: torch.Tensor  # Outcome features

class PICOParser(nn.Module):
    """Parse clinical queries into PICO structure."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Population encoder (patient features)
        self.population_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Intervention encoder (treatment features)
        self.intervention_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Comparison encoder (control features)
        self.comparison_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Outcome encoder
        self.outcome_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, patient_features: torch.Tensor, 
                treatment_features: torch.Tensor,
                control_features: torch.Tensor,
                outcome_features: torch.Tensor) -> PICOStructure:
        """Parse features into PICO structure."""
        # Ensure treatment features have correct dimension
        if treatment_features.size(1) != self.input_dim:
            # Pad treatment features to match input_dim
            padding = torch.zeros(treatment_features.size(0), 
                                self.input_dim - treatment_features.size(1),
                                device=treatment_features.device)
            treatment_features = torch.cat([treatment_features, padding], dim=1)
            control_features = torch.cat([control_features, padding], dim=1)
            outcome_features = torch.cat([outcome_features, padding], dim=1)
        
        population = self.population_encoder(patient_features)
        intervention = self.intervention_encoder(treatment_features)
        comparison = self.comparison_encoder(control_features)
        outcome = self.outcome_encoder(outcome_features)
        
        return PICOStructure(
            population=population,
            intervention=intervention,
            comparison=comparison,
            outcome=outcome
        )

class ContrastiveEmbedding(nn.Module):
    """Generate contrastive embeddings for treatment effect similarity."""
    
    def __init__(self, hidden_dim: int = 256, embedding_dim: int = 768):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Treatment effect encoder
        self.effect_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Contrastive margin
        self.margin = 0.5
    
    def forward(self, pico: PICOStructure) -> torch.Tensor:
        """
        Generate contrastive embedding: f(P,I) - f(P,C) for outcome O
        
        This captures the treatment effect similarity.
        """
        # Combine population with intervention
        pi_combined = torch.cat([pico.population, pico.intervention], dim=1)
        
        # Combine population with comparison
        pc_combined = torch.cat([pico.population, pico.comparison], dim=1)
        
        # Encode treatment effects
        pi_effect = self.effect_encoder(pi_combined)
        pc_effect = self.effect_encoder(pc_combined)
        
        # Contrastive embedding: difference captures treatment effect
        contrastive_embedding = pi_effect - pc_effect
        
        # Normalize
        contrastive_embedding = F.normalize(contrastive_embedding, dim=1)
        
        return contrastive_embedding
    
    def compute_contrastive_loss(self, anchor: torch.Tensor, 
                                positive: torch.Tensor,
                                negative: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss for treatment effect similarity."""
        # Distance between anchor and positive (same treatment effect)
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        
        # Distance between anchor and negative (different treatment effect)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        # Contrastive loss: pull positives together, push negatives apart
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0).mean()
        
        return loss

class NuanceWeightedRetrieval(nn.Module):
    """Retrieve documents with statistical nuance weighting."""
    
    def __init__(self, embedding_dim: int = 768, top_k: int = 5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.top_k = top_k
        
        # Nuance scorer (weights documents by statistical quality)
        self.nuance_scorer = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, query_embedding: torch.Tensor,
                corpus_embeddings: torch.Tensor,
                nuance_scores: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve documents with nuance weighting.
        
        Args:
            query_embedding: Contrastive query embedding [B, embedding_dim]
            corpus_embeddings: Document embeddings [N_docs, embedding_dim]
            nuance_scores: Optional pre-computed nuance scores [N_docs, 1]
        
        Returns:
            retrieved_embeddings: [B, top_k, embedding_dim]
            retrieval_scores: [B, top_k]
            retrieval_indices: [B, top_k]
        """
        # Normalize embeddings
        query_emb = F.normalize(query_embedding, dim=1)
        corpus_emb = F.normalize(corpus_embeddings, dim=1)
        
        # Compute similarity
        similarity = torch.matmul(query_emb, corpus_emb.t())  # [B, N_docs]
        
        # Compute nuance scores if not provided
        if nuance_scores is None:
            nuance_scores = self.nuance_scorer(corpus_embeddings)  # [N_docs, 1]
            nuance_scores = nuance_scores.t()  # [1, N_docs]
        
        # Weight similarity by nuance
        weighted_similarity = similarity * nuance_scores  # [B, N_docs]
        
        # Get top-k documents
        top_k = min(self.top_k, weighted_similarity.size(1))
        scores, indices = torch.topk(weighted_similarity, k=top_k, dim=1)
        
        # Gather retrieved embeddings
        batch_size = query_emb.size(0)
        expanded_indices = indices.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
        retrieved_embeddings = torch.gather(
            corpus_embeddings.unsqueeze(0).expand(batch_size, -1, -1),
            1,
            expanded_indices
        )
        
        return retrieved_embeddings, scores, indices

class PICOContrastiveRAG(nn.Module):
    """
    PICO-Contrastive RAG for drug recommendation.
    
    Key innovation: Uses contrastive embeddings for treatment effect similarity
    instead of semantic similarity.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # PICO parser
        self.pico_parser = PICOParser(
            input_dim=config.confounder_dim + config.treatment_dim,
            hidden_dim=config.hidden_dim
        )
        
        # Contrastive embedding generator
        self.contrastive_embedding = ContrastiveEmbedding(
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim
        )
        
        # Nuance-weighted retrieval
        self.retrieval = NuanceWeightedRetrieval(
            embedding_dim=config.embedding_dim,
            top_k=config.retrieval_top_k
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
        
        # Loss weights
        self.lambda_causal = 0.5
        self.lambda_retrieval = 0.3
    
    def forward(self, batch: Dict, corpus_embeddings: torch.Tensor = None) -> Dict:
        """Forward pass with PICO-contrastive retrieval."""
        patient_features = batch['patient']
        treatment = batch['treatment']
        confounders = batch['confounders']
        
        batch_size = patient_features.size(0)
        
        # Create control treatment (all zeros for now)
        control_treatment = torch.zeros_like(treatment)
        
        # Create outcome features (simplified)
        outcome_features = torch.randn_like(patient_features)
        
        # Parse PICO structure
        pico = self.pico_parser(
            patient_features,
            treatment,
            control_treatment,
            outcome_features
        )
        
        # Generate contrastive embedding
        contrastive_emb = self.contrastive_embedding(pico)
        
        # Retrieve documents with nuance weighting
        if corpus_embeddings is not None:
            retrieved_embeddings, retrieval_scores, retrieval_indices = self.retrieval(
                contrastive_emb,
                corpus_embeddings
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
        
        return {
            'outcome': outcome,
            'retrieval_scores': retrieval_scores,
            'retrieval_indices': retrieval_indices,
            'contrastive_embedding': contrastive_emb,
            'pico_structure': pico
        }
    
    def compute_loss(self, pred: Dict, batch: Dict) -> torch.Tensor:
        """Compute loss with contrastive regularization."""
        # Prediction loss
        prediction_loss = F.mse_loss(pred['outcome'], batch['outcome'])
        
        # Contrastive loss (if we have multiple treatment groups)
        if 'contrastive_embedding' in pred:
            # For now, use retrieval scores as proxy for contrastive loss
            # In real implementation, would use actual treatment effect similarity
            retrieval_scores = pred['retrieval_scores']
            contrastive_loss = -torch.log(retrieval_scores + 1e-8).mean()
        else:
            contrastive_loss = torch.tensor(0.0, device=prediction_loss.device)
        
        # Total loss
        total_loss = prediction_loss + self.lambda_causal * contrastive_loss
        
        return total_loss

def create_pico_contrastive_rag(config):
    """Factory function to create PICO-Contrastive RAG."""
    return PICOContrastiveRAG(config)

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
    model = create_pico_contrastive_rag(config)
    
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
    
    print("PICO-Contrastive RAG Test:")
    print(f"  Outcome shape: {pred['outcome'].shape}")
    print(f"  Contrastive embedding shape: {pred['contrastive_embedding'].shape}")
    print(f"  Retrieval scores shape: {pred['retrieval_scores'].shape}")
    
    # Compute loss
    loss = model.compute_loss(pred, batch)
    print(f"  Loss: {loss.item():.4f}")
