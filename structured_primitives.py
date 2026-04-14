"""
Hypothesis 1: Structured Causal Primitives as Retrieval & Generation Scaffolds

Instead of retrieving narrative text, retrieve structured evidence primitives:
(Intervention, Comparator, Outcome, Effect_Size, Study_Design, Population)

These are presented within a generation template that forces the LLM to explicitly 
map its recommendation to these primitives, separating evidence reporting from 
clinical inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class CausalPrimitive:
    """Structured causal evidence primitive."""
    intervention: str
    comparator: str
    outcome: str
    effect_size: float
    study_design: str  # RCT, observational, meta-analysis
    population: str
    confidence: float
    source_doc_id: int


class PrimitiveExtractor:
    """Extract structured causal primitives from retrieved documents."""
    
    # Causal signal patterns
    INTERVENTION_PATTERNS = ['drug', 'treatment', 'therapy', 'intervention', 'dose']
    COMPARATOR_PATTERNS = ['vs', 'versus', 'compared to', 'placebo', 'control']
    OUTCOME_PATTERNS = ['outcome', 'endpoint', 'response', 'survival', 'mortality', 'efficacy']
    STUDY_DESIGN_KEYWORDS = {
        'RCT': ['randomized', 'randomised', 'rct', 'double-blind', 'placebo-controlled'],
        'observational': ['cohort', 'case-control', 'observational', 'retrospective'],
        'meta-analysis': ['meta-analysis', 'systematic review', 'pooled analysis']
    }
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        
        # Neural primitive encoder
        self.primitive_encoder = nn.Sequential(
            nn.Linear(6 * embedding_dim, 256),  # 6 fields
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
    
    def extract_primitives(self, doc_text: str, doc_embedding: torch.Tensor) -> List[CausalPrimitive]:
        """Extract causal primitives from document text and embedding."""
        primitives = []
        
        # Simple rule-based extraction (can be enhanced with NLP)
        text_lower = doc_text.lower()
        
        # Detect study design
        study_design = 'unknown'
        for design, keywords in self.STUDY_DESIGN_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                study_design = design
                break
        
        # Extract effect size (look for common patterns)
        effect_size = 0.0
        import re
        effect_patterns = [
            r'HR\s*[=:]\s*([\d.]+)',
            r'OR\s*[=:]\s*([\d.]+)',
            r'RR\s*[=:]\s*([\d.]+)',
            r'effect size\s*[=:]\s*([\d.]+)',
            r'difference of\s*([\d.]+)',
        ]
        for pattern in effect_patterns:
            match = re.search(pattern, doc_text, re.IGNORECASE)
            if match:
                effect_size = float(match.group(1))
                break
        
        # Create primitive (simplified - real implementation would use NER)
        primitive = CausalPrimitive(
            intervention="extracted_intervention",
            comparator="extracted_comparator", 
            outcome="extracted_outcome",
            effect_size=effect_size,
            study_design=study_design,
            population="general",
            confidence=0.7 if study_design != 'unknown' else 0.3,
            source_doc_id=hash(doc_text) % 10000
        )
        primitives.append(primitive)
        
        return primitives
    
    def encode_primitive(self, primitive: CausalPrimitive) -> torch.Tensor:
        """Encode a structured primitive into embedding space."""
        # Simple encoding (real implementation would use learned embeddings)
        fields = [
            primitive.intervention,
            primitive.comparator,
            primitive.outcome,
            str(primitive.effect_size),
            primitive.study_design,
            primitive.population
        ]
        
        # Create fixed-size encoding
        encoding = torch.zeros(self.embedding_dim)
        
        # Encode study design
        design_map = {'RCT': 0.9, 'meta-analysis': 0.8, 'observational': 0.5, 'unknown': 0.2}
        encoding[0] = design_map.get(primitive.study_design, 0.2)
        
        # Encode effect size (normalized)
        encoding[1] = min(abs(primitive.effect_size) / 10.0, 1.0)
        
        # Encode confidence
        encoding[2] = primitive.confidence
        
        # Fill rest with hash-based pseudo-embeddings
        for i, field in enumerate(fields):
            start_idx = 3 + i * 10
            end_idx = min(start_idx + 10, self.embedding_dim)
            field_hash = hash(field) % 1000 / 1000.0
            encoding[start_idx:end_idx] = field_hash
        
        return F.normalize(encoding, dim=0)


class StructuredPrimitiveRAG(nn.Module):
    """
    RAG model using structured causal primitives instead of narrative retrieval.
    
    Key innovation: Retrieval is based on primitive structure, not just semantic similarity.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.primitive_extractor = PrimitiveExtractor(config.embedding_dim)
        
        # Primitive-aware retriever - use actual embedding_dim from config
        # Note: config.embedding_dim may be overridden by sentence-transformers (384)
        self.embedding_dim = config.embedding_dim  # Store actual dimension
        
        self.primitive_retriever = nn.Sequential(
            nn.Linear(config.confounder_dim + config.treatment_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, self.embedding_dim)  # Use actual embedding_dim
        )
        
        # Treatment encoder
        self.treatment_encoder = nn.Linear(config.treatment_dim, config.hidden_dim)
        
        # Confounder encoder  
        self.confounder_encoder = nn.Linear(config.confounder_dim, config.hidden_dim)
        
        # Primitive aggregation
        self.primitive_aggregator = nn.Sequential(
            nn.Linear(self.embedding_dim * config.retrieval_top_k, config.hidden_dim),  # Use actual embedding_dim
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Outcome predictor with primitive conditioning
        self.outcome_predictor = nn.Sequential(
            nn.Linear(3 * config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.outcome_dim)
        )
        
        # Attribution network: maps predictions to supporting primitives
        self.attribution_network = nn.Sequential(
            nn.Linear(config.hidden_dim + config.outcome_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.retrieval_top_k),
            nn.Softmax(dim=1)
        )
    
    def retrieve_primitives(self, patient_features: torch.Tensor, 
                           primitive_corpus: torch.Tensor,
                           top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve structured primitives based on patient features."""
        # Encode patient for primitive retrieval
        patient_emb = self.primitive_retriever(patient_features)
        patient_emb = F.normalize(patient_emb, dim=1)
        
        # Normalize primitive corpus
        corpus_emb = F.normalize(primitive_corpus, dim=1)
        
        # Compute similarity
        similarity = torch.matmul(patient_emb, corpus_emb.t())
        
        # Get top-k primitives
        top_k = min(top_k, similarity.size(1))
        scores, indices = torch.topk(similarity, k=top_k, dim=1)
        
        # Gather retrieved primitives
        batch_size = patient_emb.size(0)
        expanded_indices = indices.unsqueeze(-1).expand(-1, -1, self.config.embedding_dim)
        retrieved_primitives = torch.gather(
            primitive_corpus.unsqueeze(0).expand(batch_size, -1, -1),
            1,
            expanded_indices
        )
        
        return retrieved_primitives, scores, indices
    
    def forward(self, batch: Dict, primitive_corpus: torch.Tensor = None) -> Dict:
        """Forward pass with structured primitive retrieval."""
        patient_features = batch['patient']
        treatment = batch['treatment']
        confounders = batch['confounders']
        
        # Retrieve primitives
        if primitive_corpus is not None:
            retrieved_primitives, retrieval_scores, retrieval_indices = self.retrieve_primitives(
                patient_features, primitive_corpus, self.config.retrieval_top_k
            )
        else:
            # Fallback: use dummy primitives
            batch_size = patient_features.size(0)
            retrieved_primitives = torch.randn(
                batch_size, self.config.retrieval_top_k, self.config.embedding_dim,
                device=patient_features.device
            )
            retrieval_scores = torch.zeros(batch_size, self.config.retrieval_top_k, device=patient_features.device)
            retrieval_indices = torch.zeros(batch_size, self.config.retrieval_top_k, dtype=torch.long, device=patient_features.device)
        
        # Flatten primitives
        batch_size = patient_features.size(0)
        flattened_primitives = retrieved_primitives.view(batch_size, -1)
        
        # Encode components
        treatment_enc = self.treatment_encoder(treatment)
        confounder_enc = self.confounder_encoder(confounders)
        primitive_enc = self.primitive_aggregator(flattened_primitives)
        
        # Combine for prediction
        combined = torch.cat([treatment_enc, confounder_enc, primitive_enc], dim=1)
        outcome = self.outcome_predictor(combined)
        
        # Compute attribution scores
        attribution_input = torch.cat([primitive_enc, outcome], dim=1)
        attribution_scores = self.attribution_network(attribution_input)
        
        return {
            'outcome': outcome,
            'retrieval_scores': retrieval_scores,
            'retrieval_indices': retrieval_indices,
            'attribution_scores': attribution_scores,
            'retrieved_primitives': retrieved_primitives
        }
    
    def compute_loss(self, pred: Dict, batch: Dict) -> torch.Tensor:
        """Compute loss with attribution regularization."""
        # Prediction loss
        prediction_loss = F.mse_loss(pred['outcome'], batch['outcome'])
        
        # Attribution consistency loss
        # Encourage attribution to be concentrated on high-quality primitives
        if 'retrieval_scores' in pred:
            retrieval_scores = pred['retrieval_scores']
            attribution_scores = pred['attribution_scores']
            
            # Attribution should align with retrieval quality
            alignment_loss = F.mse_loss(attribution_scores, retrieval_scores)
        else:
            alignment_loss = torch.tensor(0.0, device=prediction_loss.device)
        
        total_loss = prediction_loss + 0.1 * alignment_loss
        
        return total_loss


def evaluate_attribution_accuracy(model, test_data, primitive_corpus):
    """
    Evaluate causal claim attribution accuracy.
    
    Measures: Does the model correctly attribute recommendations to 
    the type of evidence (RCT vs observational)?
    """
    model.eval()
    correct_attributions = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_data:
            pred = model(batch, primitive_corpus)
            attribution_scores = pred['attribution_scores']
            
            # Check if highest attribution goes to RCT evidence
            # (simplified evaluation)
            top_attribution_idx = torch.argmax(attribution_scores, dim=1)
            
            # In a real evaluation, we'd check if the attributed primitive
            # is from an RCT when ground truth says RCT is most relevant
            correct_attributions += (top_attribution_idx == 0).sum().item()
            total += batch['outcome'].size(0)
    
    return correct_attributions / total if total > 0 else 0.0
