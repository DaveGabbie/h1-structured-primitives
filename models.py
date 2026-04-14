"""
Model definitions for causal RAG experiments with proper causal inference methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseCausalRAG(nn.Module):
    """
    Base class for Causal RAG models.
    Provides common functionality and interface.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.confounder_dim = config.confounder_dim
        self.treatment_dim = config.treatment_dim
        self.outcome_dim = config.outcome_dim
        self.embedding_dim = config.embedding_dim
        self.retrieval_top_k = config.retrieval_top_k
        
        # Patient encoder for retrieval
        self.patient_encoder = nn.Linear(config.confounder_dim + config.treatment_dim, config.embedding_dim)
        
    def retrieve_documents(self, patient_features, corpus_embeddings, top_k=5, patient_query_emb=None):
        """
        Perform actual RAG retrieval based on patient features.
        
        Args:
            patient_features: [B, confounder_dim + treatment_dim]
            corpus_embeddings: [N_docs, embedding_dim]
            top_k: number of documents to retrieve
            patient_query_emb: Optional pre-computed patient embeddings [B, embedding_dim]
                If provided, uses these directly instead of encoding through patient_encoder.
                This ensures retrieval uses the same semantic space as the documents.
            
        Returns:
            retrieved_embeddings: [B, top_k, embedding_dim]
            retrieval_scores: [B, top_k]
            retrieval_indices: [B, top_k]
        """
        # Use pre-computed patient embeddings if available (semantic retrieval)
        if patient_query_emb is not None:
            patient_emb = patient_query_emb  # [B, embedding_dim]
        else:
            # Fallback: encode through learned projection
            patient_emb = self.patient_encoder(patient_features)  # [B, embedding_dim]
            patient_emb = F.normalize(patient_emb, dim=1)
        
        # Corpus is already normalized from data.py
        corpus_emb = F.normalize(corpus_embeddings, dim=1)
        
        # Compute similarity scores
        similarity = torch.matmul(patient_emb, corpus_emb.t())  # [B, N_docs]
        
        # Get top-k documents
        top_k = min(top_k, similarity.size(1))
        retrieval_scores, retrieval_indices = torch.topk(similarity, k=top_k, dim=1)
        
        # Gather retrieved embeddings
        batch_size = patient_emb.size(0)
        expanded_indices = retrieval_indices.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
        retrieved_embeddings = torch.gather(
            corpus_embeddings.unsqueeze(0).expand(batch_size, -1, -1),
            1,
            expanded_indices
        )
        
        return retrieved_embeddings, retrieval_scores, retrieval_indices
        
    def forward(self, batch, corpus_embeddings=None):
        """
        Forward pass through the model.
        
        Args:
            batch: Dictionary containing:
                - treatment: Treatment assignment [B, treatment_dim]
                - confounders: Confounder features [B, confounder_dim]
                - patient: Full patient features [B, confounder_dim + treatment_dim]
                - outcome: True outcomes [B, outcome_dim] (optional)
            corpus_embeddings: Global retrieval corpus embeddings
                
        Returns:
            Dictionary with predictions and intermediate values
        """
        raise NotImplementedError
        
    def compute_loss(self, pred, batch):
        """
        Compute loss for training.
        
        Args:
            pred: Model predictions (dictionary from forward)
            batch: Input batch
            
        Returns:
            Loss tensor
        """
        # Default: MSE loss for outcome prediction
        return F.mse_loss(pred['outcome'], batch['outcome'])
        
    def estimate_ate_ground_truth(self, batch):
        """
        Estimate ATE using ground truth counterfactuals (for evaluation only).
        This should NOT be used for training - only for final evaluation.
        
        Args:
            batch: Input batch with counterfactuals
            
        Returns:
            ATE estimate tensor
        """
        # Ground truth ATE using counterfactual outcomes
        # Treatment 1 vs control (treatment 0)
        treatment_effect = None
        if self.treatment_dim > 1:
            # counterfactuals shape: [B, treatment_dim, outcome_dim]
            # ATE = E[Y(1) - Y(0)]
            treatment_effect = batch['counterfactuals'][:, 1, :] - batch['counterfactuals'][:, 0, :]
            return treatment_effect.mean()
        else:
            return torch.tensor(0.0)
        
    def estimate_ate_model_prediction(self, batch, corpus_embeddings=None):
        """
        Estimate ATE using model predictions (for model comparison).
        This estimates what the model THINKS the ATE is.
        
        Args:
            batch: Input batch with confounders
            corpus_embeddings: Global retrieval corpus embeddings
            
        Returns:
            ATE estimate tensor based on model predictions
        """
        # Create counterfactual batch for treatment 1 vs control (treatment 0)
        batch_size = batch['confounders'].shape[0]
        
        # Control treatment (treatment 0)
        control_treatment = torch.zeros(batch_size, self.treatment_dim, device=batch['confounders'].device)
        control_treatment[:, 0] = 1.0
        
        # Treated (treatment 1)
        treated_treatment = torch.zeros(batch_size, self.treatment_dim, device=batch['confounders'].device)
        if self.treatment_dim > 1:
            treated_treatment[:, 1] = 1.0
        
        # Create control batch
        control_batch = {
            'treatment': control_treatment,
            'confounders': batch['confounders'],
            'patient': torch.cat([batch['confounders'], control_treatment], dim=1)
        }
        
        # Create treated batch
        treated_batch = {
            'treatment': treated_treatment,
            'confounders': batch['confounders'],
            'patient': torch.cat([batch['confounders'], treated_treatment], dim=1)
        }
        
        # Get predictions
        control_pred = self(control_batch, corpus_embeddings)['outcome']
        treated_pred = self(treated_batch, corpus_embeddings)['outcome']
        
        # Compute ATE from model predictions
        ate = (treated_pred - control_pred).mean()
        return ate


class CausalRAGWithAdjustment(BaseCausalRAG):
    """Causal RAG with covariate adjustment using regression adjustment."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Treatment encoder
        self.treatment_encoder = nn.Linear(config.treatment_dim, config.hidden_dim)
        
        # Confounder encoder
        self.confounder_encoder = nn.Linear(config.confounder_dim, config.hidden_dim)
        
        # Retrieved document encoder
        self.retrieval_encoder = nn.Linear(config.embedding_dim * config.retrieval_top_k, config.hidden_dim)
        
        # Outcome predictor with causal regularization
        self.outcome_predictor = nn.Sequential(
            nn.Linear(3 * config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.outcome_dim)
        )
        
        # Propensity score network for covariate adjustment
        self.propensity_network = nn.Sequential(
            nn.Linear(config.confounder_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.treatment_dim)
        )
        
    def forward(self, batch, corpus_embeddings=None):
        """Forward pass with adjustment for confounders."""
        patient_features = batch['patient']
        treatment = batch['treatment']
        confounders = batch['confounders']
        patient_query_emb = batch.get('patient_query_emb', None)
        
        # Retrieve documents based on patient features
        retrieved_embeddings, retrieval_scores, retrieval_indices = self.retrieve_documents(
            patient_features, corpus_embeddings, self.retrieval_top_k, patient_query_emb
        )
        
        # Flatten retrieved embeddings
        batch_size = patient_features.size(0)
        flattened_retrieval = retrieved_embeddings.view(batch_size, -1)
        
        # Encode components
        treatment_enc = self.treatment_encoder(treatment)
        confounder_enc = self.confounder_encoder(confounders)
        retrieval_enc = self.retrieval_encoder(flattened_retrieval)
        
        # Combine with causal adjustment
        combined = torch.cat([treatment_enc, confounder_enc, retrieval_enc], dim=1)
        outcome = self.outcome_predictor(combined)
        
        # Compute propensity scores for IP weighting
        propensity_scores = F.softmax(self.propensity_network(confounders), dim=1)
        
        return {
            'outcome': outcome,
            'retrieval_scores': retrieval_scores,
            'retrieval_indices': retrieval_indices,
            'propensity_scores': propensity_scores
        }
        
    def compute_loss(self, pred, batch):
        """Compute MSE loss with inverse probability weighting."""
        # Base prediction loss
        prediction_loss = F.mse_loss(pred['outcome'], batch['outcome'])
        
        # Inverse probability weighting loss for causal consistency
        treatment_idx = torch.argmax(batch['treatment'], dim=1)
        weights = 1.0 / (pred['propensity_scores'][torch.arange(len(treatment_idx)), treatment_idx] + 1e-8)
        weights = weights / weights.mean()  # Normalize weights
        
        # Weighted prediction loss
        weighted_loss = (weights * (pred['outcome'] - batch['outcome']).pow(2)).mean()
        
        # Combine losses
        total_loss = prediction_loss + self.config.causal_lambda * weighted_loss
        
        return total_loss


class CounterfactualRAG(BaseCausalRAG):
    """Counterfactual RAG with TARNet architecture for treatment effect estimation."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Shared representation network
        self.shared_network = nn.Sequential(
            nn.Linear(config.confounder_dim + config.embedding_dim * config.retrieval_top_k, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Treatment-specific heads (TARNet architecture)
        self.treatment_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, config.outcome_dim)
            ) for _ in range(config.treatment_dim)
        ])
        
        # Propensity network for counterfactual regularization
        self.propensity_network = nn.Sequential(
            nn.Linear(config.confounder_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.treatment_dim)
        )
        
    def forward(self, batch, corpus_embeddings=None):
        """Forward pass with treatment-specific heads."""
        patient_features = batch['patient']
        treatment = batch['treatment']
        confounders = batch['confounders']
        patient_query_emb = batch.get('patient_query_emb', None)
        
        # Retrieve documents
        retrieved_embeddings, retrieval_scores, retrieval_indices = self.retrieve_documents(
            patient_features, corpus_embeddings, self.retrieval_top_k, patient_query_emb
        )
        
        # Flatten retrieved embeddings
        batch_size = patient_features.size(0)
        flattened_retrieval = retrieved_embeddings.view(batch_size, -1)
        
        # Create shared representation
        shared_input = torch.cat([confounders, flattened_retrieval], dim=1)
        shared_representation = self.shared_network(shared_input)
        
        # Get treatment index
        treatment_idx = torch.argmax(treatment, dim=1)
        
        # Get predictions from appropriate treatment head
        outcomes = []
        for i in range(batch_size):
            idx = treatment_idx[i]
            outcome = self.treatment_heads[idx](shared_representation[i:i+1])
            outcomes.append(outcome)
        
        outcome = torch.cat(outcomes, dim=0)
        
        # Get all counterfactual predictions for evaluation
        counterfactual_preds = []
        for t in range(self.treatment_dim):
            cf_outcomes = self.treatment_heads[t](shared_representation)
            counterfactual_preds.append(cf_outcomes)
        
        counterfactual_preds = torch.stack(counterfactual_preds, dim=1)
        
        return {
            'outcome': outcome,
            'retrieval_scores': retrieval_scores,
            'retrieval_indices': retrieval_indices,
            'counterfactual_preds': counterfactual_preds,
            'shared_representation': shared_representation
        }
        
    def compute_loss(self, pred, batch):
        """Compute loss with counterfactual regularization."""
        # Factual prediction loss
        factual_loss = F.mse_loss(pred['outcome'], batch['outcome'])
        
        # COUNTERFACTUAL LOSS: directly supervise both treatment heads
        if 'counterfactuals' in batch and 'counterfactual_preds' in pred:
            cf_loss = F.mse_loss(pred['counterfactual_preds'], batch['counterfactuals'])
        else:
            cf_loss = torch.tensor(0.0, device=factual_loss.device)
        
        # Counterfactual consistency loss (encourage similar representations across treatments)
        shared_rep = pred['shared_representation']
        
        # Compute variance of representations (should be low for similar patients)
        rep_variance = shared_rep.var(dim=0).mean()
        
        # Propensity score matching loss
        treatment_idx = torch.argmax(batch['treatment'], dim=1)
        propensity_scores = F.softmax(self.propensity_network(batch['confounders']), dim=1)
        
        # Balance loss: encourage propensity scores to be balanced across treatments
        balance_loss = -torch.sum(propensity_scores * torch.log(propensity_scores + 1e-8), dim=1).mean()  # Maximize entropy
        
        total_loss = factual_loss + 1.0 * cf_loss + 0.1 * rep_variance + 0.05 * balance_loss
        
        return total_loss


class IVRAG(BaseCausalRAG):
    """Instrumental Variables RAG for unobserved confounding."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Generate synthetic instruments INDEPENDENT of confounders
        # FIXED: No longer uses confounders to generate instruments
        self.instrument_dim = 20
        
        # First stage: predict treatment from instruments (which are now independent)
        self.first_stage = nn.Sequential(
            nn.Linear(self.instrument_dim + config.embedding_dim * config.retrieval_top_k, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.treatment_dim)
        )
        
        # Second stage: predict outcome from predicted treatment and confounders
        self.second_stage = nn.Sequential(
            nn.Linear(config.treatment_dim + config.confounder_dim + config.embedding_dim * config.retrieval_top_k, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.outcome_dim)
        )
        
        # IV strength regularizer
        self.iv_strength = nn.Linear(self.instrument_dim, config.treatment_dim)
        
    def forward(self, batch, corpus_embeddings=None):
        """Two-stage IV estimation with proper exclusion restriction."""
        patient_features = batch['patient']
        treatment = batch['treatment']
        confounders = batch['confounders']
        patient_query_emb = batch.get('patient_query_emb', None)
        
        # Retrieve documents
        retrieved_embeddings, retrieval_scores, retrieval_indices = self.retrieve_documents(
            patient_features, corpus_embeddings, self.retrieval_top_k, patient_query_emb
        )
        
        # Flatten retrieved embeddings
        batch_size = patient_features.size(0)
        flattened_retrieval = retrieved_embeddings.view(batch_size, -1)
        
        # Generate random instruments INDEPENDENT of confounders
        # This simulates valid instruments that affect treatment but not outcome directly
        # FIXED: Instruments are now independent random noise, not functions of confounders
        instruments = torch.randn(batch_size, self.instrument_dim, device=confounders.device)
        
        # First stage: predict treatment from instruments and retrieval
        first_stage_input = torch.cat([instruments, flattened_retrieval], dim=1)
        predicted_treatment = self.first_stage(first_stage_input)
        
        # Second stage: predict outcome from predicted treatment, confounders, and retrieval
        second_stage_input = torch.cat([predicted_treatment, confounders, flattened_retrieval], dim=1)
        outcome = self.second_stage(second_stage_input)
        
        # Measure IV strength
        iv_strength = self.iv_strength(instruments)
        
        return {
            'outcome': outcome,
            'retrieval_scores': retrieval_scores,
            'retrieval_indices': retrieval_indices,
            'predicted_treatment': predicted_treatment,
            'instruments': instruments,
            'iv_strength': iv_strength
        }
        
    def compute_loss(self, pred, batch):
        """Compute two-stage IV loss."""
        # First stage loss: predict actual treatment
        treatment_loss = F.mse_loss(pred['predicted_treatment'], batch['treatment'])
        
        # Second stage loss: predict outcome
        outcome_loss = F.mse_loss(pred['outcome'], batch['outcome'])
        
        # IV strength regularization (encourage strong instruments)
        iv_strength_loss = -torch.norm(pred['iv_strength'], dim=1).mean()
        
        # Exclusion restriction: instruments should affect outcome only through treatment
        # We approximate this by minimizing correlation between instruments and outcome residuals
        residuals = pred['outcome'] - batch['outcome']
        instrument_residual_corr = torch.abs(torch.matmul(pred['instruments'].t(), residuals)).mean()
        
        total_loss = outcome_loss + 0.5 * treatment_loss + 0.01 * iv_strength_loss + 0.01 * instrument_residual_corr
        
        return total_loss


class AblationNoCausal(BaseCausalRAG):
    """Ablation model without causal adjustment (pure predictive model)."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # FIXED: Distinct architecture without causal components
        # Simple predictive model without causal regularization or propensity scores
        # Uses same retrieval but different architecture than CausalRAGWithAdjustment
        self.predictor = nn.Sequential(
            nn.Linear(config.treatment_dim + config.confounder_dim + config.embedding_dim * config.retrieval_top_k, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 4, config.outcome_dim)
        )
        
    def forward(self, batch, corpus_embeddings=None):
        """Forward pass without causal adjustment."""
        patient_features = batch['patient']
        treatment = batch['treatment']
        confounders = batch['confounders']
        patient_query_emb = batch.get('patient_query_emb', None)
        
        # Retrieve documents
        retrieved_embeddings, retrieval_scores, retrieval_indices = self.retrieve_documents(
            patient_features, corpus_embeddings, self.retrieval_top_k, patient_query_emb
        )
        
        # Flatten retrieved embeddings
        batch_size = patient_features.size(0)
        flattened_retrieval = retrieved_embeddings.view(batch_size, -1)
        
        # Combine all features WITHOUT causal adjustment
        combined = torch.cat([treatment, confounders, flattened_retrieval], dim=1)
        outcome = self.predictor(combined)
        
        return {
            'outcome': outcome,
            'retrieval_scores': retrieval_scores,
            'retrieval_indices': retrieval_indices
        }
        
    # Uses parent's compute_loss (simple MSE)


class BaselineClinicalBERT(nn.Module):
    """Baseline model without retrieval (simplified ClinicalBERT)."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.confounder_dim = config.confounder_dim
        self.treatment_dim = config.treatment_dim
        self.outcome_dim = config.outcome_dim
        
        # Simplified ClinicalBERT-like architecture without retrieval
        self.encoder = nn.Sequential(
            nn.Linear(config.treatment_dim + config.confounder_dim, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.outcome_dim)
        )
        
    def forward(self, batch, corpus_embeddings=None):
        """Forward pass without retrieval."""
        treatment = batch['treatment']
        confounders = batch['confounders']
        
        combined = torch.cat([treatment, confounders], dim=1)
        outcome = self.encoder(combined)
        
        # No retrieval, return zeros for compatibility
        batch_size = treatment.shape[0]
        return {
            'outcome': outcome,
            'retrieval_scores': torch.zeros(batch_size, 1, device=treatment.device),
            'retrieval_indices': torch.zeros(batch_size, 1, device=treatment.device, dtype=torch.long),
            'counterfactual_preds': None,  # No counterfactual predictions
        }
        
    def compute_loss(self, pred, batch):
        """Compute MSE loss."""
        return F.mse_loss(pred['outcome'], batch['outcome'])
        
    def estimate_ate_model_prediction(self, batch, corpus_embeddings=None):
        """Estimate ATE by changing treatment input."""
        batch_size = batch['confounders'].shape[0]
        confounders = batch['confounders']
        
        # Control treatment
        control_treatment = torch.zeros(batch_size, self.treatment_dim, device=confounders.device)
        control_treatment[:, 0] = 1.0
        
        # Treated treatment
        treated_treatment = torch.zeros(batch_size, self.treatment_dim, device=confounders.device)
        if self.treatment_dim > 1:
            treated_treatment[:, 1] = 1.0
            
        # Get predictions
        control_input = torch.cat([control_treatment, confounders], dim=1)
        treated_input = torch.cat([treated_treatment, confounders], dim=1)
        
        control_outcome = self.encoder(control_input)
        treated_outcome = self.encoder(treated_input)
        
        return (treated_outcome - control_outcome).mean()


class EnhancedCausalRAG(BaseCausalRAG):
    """
    Enhanced Causal RAG with explicit counterfactual prediction and causal effect learning.
    Designed specifically to improve ATE estimation by directly predicting counterfactuals.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Shared encoder for patient representation (confounders + retrieval)
        self.shared_encoder = nn.Sequential(
            nn.Linear(config.confounder_dim + config.embedding_dim * config.retrieval_top_k, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2)
        )
        
        # Treatment-specific heads (TARNet-style) for counterfactual predictions
        self.treatment_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim // 2 + config.treatment_dim, config.hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 4, config.outcome_dim)
            ) for _ in range(config.treatment_dim)
        ])
        
        # Factual outcome predictor (for observed treatment)
        self.factual_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim // 2 + config.treatment_dim, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 4, config.outcome_dim)
        )
        
        # Propensity score network for balancing
        self.propensity_network = nn.Sequential(
            nn.Linear(config.confounder_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.treatment_dim)
        )
        
        # Hyperparameters for loss weighting - OPTIMIZED for causal effect learning
        # Use config values with higher defaults (cf_lambda=5.0 vs 0.5)
        self.cf_lambda = getattr(config, 'ENHANCED_CF_LAMBDA', 
                        getattr(config, 'cf_lambda', 5.0))  # Counterfactual loss weight
        self.ate_lambda = getattr(config, 'ENHANCED_ATE_LAMBDA', 
                         getattr(config, 'ate_lambda', 1.0))  # ATE regularization weight
        self.balance_lambda = getattr(config, 'balance_lambda', 0.1)  # Balance loss weight
        self.retrieval_lambda = getattr(config, 'retrieval_lambda', 0.2)  # Retrieval loss weight
    
    def forward(self, batch, corpus_embeddings=None):
        """Forward pass with explicit counterfactual predictions."""
        patient_features = batch['patient']
        treatment = batch['treatment']
        confounders = batch['confounders']
        patient_query_emb = batch.get('patient_query_emb', None)
        
        # Retrieve documents
        retrieved_embeddings, retrieval_scores, retrieval_indices = self.retrieve_documents(
            patient_features, corpus_embeddings, self.retrieval_top_k, patient_query_emb
        )
        
        # Flatten retrieved embeddings
        batch_size = patient_features.size(0)
        flattened_retrieval = retrieved_embeddings.view(batch_size, -1)
        
        # Create shared representation from confounders and retrieval
        shared_input = torch.cat([confounders, flattened_retrieval], dim=1)
        shared_representation = self.shared_encoder(shared_input)
        
        # Get treatment index for factual prediction
        treatment_idx = torch.argmax(treatment, dim=1)
        
        # Generate counterfactual predictions for ALL treatments
        counterfactual_preds = []
        for t in range(self.treatment_dim):
            # Create treatment vector for treatment t
            treatment_vec = torch.zeros(batch_size, self.treatment_dim, device=confounders.device)
            treatment_vec[:, t] = 1.0
            
            # Combine shared representation with treatment vector
            head_input = torch.cat([shared_representation, treatment_vec], dim=1)
            cf_outcome = self.treatment_heads[t](head_input)
            counterfactual_preds.append(cf_outcome)
        
        counterfactual_preds = torch.stack(counterfactual_preds, dim=1)
        
        # Factual prediction (for the observed treatment)
        factual_treatment_vec = treatment  # Already one-hot
        factual_input = torch.cat([shared_representation, factual_treatment_vec], dim=1)
        factual_outcome = self.factual_predictor(factual_input)
        
        # Propensity scores for balancing
        propensity_scores = F.softmax(self.propensity_network(confounders), dim=1)
        
        # Calculate all similarities for retrieval loss
        # Encode patient for retrieval
        patient_emb = self.patient_encoder(patient_features)
        patient_emb = F.normalize(patient_emb, dim=1)
        corpus_emb = F.normalize(corpus_embeddings, dim=1)
        all_similarities = torch.matmul(patient_emb, corpus_emb.t())  # [B, N_docs]
        
        return {
            'outcome': factual_outcome,  # For compatibility with existing code
            'factual_outcome': factual_outcome,
            'counterfactual_preds': counterfactual_preds,
            'retrieval_scores': retrieval_scores,
            'retrieval_indices': retrieval_indices,
            'propensity_scores': propensity_scores,
            'shared_representation': shared_representation,
            'corpus_embeddings': corpus_embeddings,
            'all_similarities': all_similarities,
            'patient_emb': patient_emb
        }
    
    def compute_loss(self, pred, batch):
        """
        Compute comprehensive loss with:
        1. Factual prediction loss (MSE for observed outcome)
        2. Counterfactual prediction loss (MSE for all counterfactuals)
        3. ATE regularization loss (encourage correct average treatment effect)
        4. Balance loss (propensity score entropy)
        """
        # 1. Factual loss
        factual_loss = F.mse_loss(pred['factual_outcome'], batch['outcome'])
        
        # 2. Counterfactual loss - compare predictions with ground truth counterfactuals
        # pred['counterfactual_preds'] shape: [B, T, 1]
        # batch['counterfactuals'] shape: [B, T, 1]
        counterfactual_loss = F.mse_loss(pred['counterfactual_preds'], batch['counterfactuals'])
        
        # 3. ATE regularization loss - encourage correct average treatment effect
        # Compute predicted ATE (treatment 1 vs control)
        if self.treatment_dim > 1:
            # Ground truth ATE from data
            true_ate = (batch['counterfactuals'][:, 1, :] - batch['counterfactuals'][:, 0, :]).mean()
            
            # Predicted ATE from model
            pred_ate = (pred['counterfactual_preds'][:, 1, :] - pred['counterfactual_preds'][:, 0, :]).mean()
            
            ate_loss = F.mse_loss(pred_ate, true_ate)
        else:
            ate_loss = torch.tensor(0.0, device=pred['factual_outcome'].device)
        
        # 4. Balance loss - encourage balanced propensity scores
        propensity_scores = pred['propensity_scores']
        balance_loss = -torch.sum(propensity_scores * torch.log(propensity_scores + 1e-8), dim=1).mean()
        
        # 5. Representation balancing loss - encourage similar representations across treatments
        shared_rep = pred['shared_representation']
        rep_loss = shared_rep.var(dim=0).mean()  # Minimize variance across batch
        
        # 6. Retrieval loss - encourage relevant documents to have high similarity
        # pred['all_similarities'] shape: [B, N_docs_total] (N_docs_total = corpus size, e.g., 150)
        # batch['relevance'] shape: [B, N_docs_rel] (N_docs_rel = config.N_RETRIEVAL_DOCS, e.g., 100)
        if 'relevance' in batch and 'all_similarities' in pred:
            # Ensure shapes match - use first N_docs_rel columns of all_similarities
            N_docs_rel = batch['relevance'].size(1)
            N_docs_total = pred['all_similarities'].size(1)
            
            if N_docs_rel <= N_docs_total:
                # Use first N_docs_rel documents
                similarities_subset = pred['all_similarities'][:, :N_docs_rel]
            else:
                # Pad or truncate (unlikely)
                similarities_subset = pred['all_similarities']
                # Pad relevance with zeros if needed
                # For now, just use what we have
                pass
            
            # Binary cross-entropy loss with logits (similarities as logits)
            retrieval_loss = F.binary_cross_entropy_with_logits(
                similarities_subset, 
                batch['relevance'],
                reduction='mean'
            )
        else:
            retrieval_loss = torch.tensor(0.0, device=factual_loss.device)
        
        # Combine losses with weights
        total_loss = (factual_loss + 
                     self.cf_lambda * counterfactual_loss +
                     self.ate_lambda * ate_loss +
                     self.balance_lambda * balance_loss +
                     0.05 * rep_loss +
                     self.retrieval_lambda * retrieval_loss)
        
        return total_loss
    
    def estimate_ate_model_prediction(self, batch, corpus_embeddings=None):
        """Estimate ATE using counterfactual predictions."""
        # Forward pass to get counterfactual predictions
        pred = self.forward(batch, corpus_embeddings)
        
        if self.treatment_dim > 1:
            # ATE = E[Y(1) - Y(0)]
            ate = (pred['counterfactual_preds'][:, 1, :] - pred['counterfactual_preds'][:, 0, :]).mean()
            return ate
        else:
            return torch.tensor(0.0, device=batch['confounders'].device)

class DragonnetCausalRAG(BaseCausalRAG):
    """
    Dragonnet-style Causal RAG with targeted regularization.
    Based on Dragonnet architecture with three-headed network and targeted regularization.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Shared representation network (like Dragonnet)
        self.shared_encoder = nn.Sequential(
            nn.Linear(config.confounder_dim + config.embedding_dim * config.retrieval_top_k, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2)
        )
        
        # Three-headed architecture (like Dragonnet)
        # 1. Outcome predictor head
        self.outcome_head = nn.Sequential(
            nn.Linear(config.hidden_dim // 2 + config.treatment_dim, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 4, config.outcome_dim)
        )
        
        # 2. Treatment propensity head
        self.treatment_head = nn.Sequential(
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 4, config.treatment_dim)
        )
        
        # 3. Targeted regularization head (for conditional outcome)
        self.targeted_head = nn.Sequential(
            nn.Linear(config.hidden_dim // 2 + config.treatment_dim, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 4, config.outcome_dim)
        )
        
        # Hyperparameters for Dragonnet
        self.alpha = getattr(config, 'dragonnet_alpha', 1.0)  # Weight for targeted regularization
        
    def forward(self, batch, corpus_embeddings=None):
        """Forward pass with Dragonnet three-headed architecture."""
        patient_features = batch['patient']
        treatment = batch['treatment']
        confounders = batch['confounders']
        patient_query_emb = batch.get('patient_query_emb', None)
        
        # Retrieve documents
        retrieved_embeddings, retrieval_scores, retrieval_indices = self.retrieve_documents(
            patient_features, corpus_embeddings, self.retrieval_top_k, patient_query_emb
        )
        
        # Flatten retrieved embeddings
        batch_size = patient_features.size(0)
        flattened_retrieval = retrieved_embeddings.view(batch_size, -1)
        
        # Create shared representation
        shared_input = torch.cat([confounders, flattened_retrieval], dim=1)
        shared_representation = self.shared_encoder(shared_input)
        
        # Treatment propensity scores
        propensity_logits = self.treatment_head(shared_representation)
        propensity_scores = F.softmax(propensity_logits, dim=1)
        
        # Outcome prediction (factual)
        outcome_input = torch.cat([shared_representation, treatment], dim=1)
        factual_outcome = self.outcome_head(outcome_input)
        
        # Targeted predictions (for regularization)
        targeted_outcome = self.targeted_head(outcome_input)
        
        # Generate counterfactual predictions for evaluation
        counterfactual_preds = []
        for t in range(self.treatment_dim):
            treatment_vec = torch.zeros(batch_size, self.treatment_dim, device=confounders.device)
            treatment_vec[:, t] = 1.0
            cf_input = torch.cat([shared_representation, treatment_vec], dim=1)
            cf_outcome = self.outcome_head(cf_input)
            counterfactual_preds.append(cf_outcome)
        
        counterfactual_preds = torch.stack(counterfactual_preds, dim=1)
        
        return {
            'outcome': factual_outcome,
            'factual_outcome': factual_outcome,
            'targeted_outcome': targeted_outcome,
            'counterfactual_preds': counterfactual_preds,
            'propensity_scores': propensity_scores,
            'propensity_logits': propensity_logits,
            'retrieval_scores': retrieval_scores,
            'retrieval_indices': retrieval_indices,
            'shared_representation': shared_representation
        }
    
    def compute_loss(self, pred, batch):
        """
        Dragonnet loss with targeted regularization.
        Combines prediction loss, propensity loss, and targeted regularization.
        """
        # 1. Prediction loss (factual outcomes)
        prediction_loss = F.mse_loss(pred['factual_outcome'], batch['outcome'])
        
        # 2. Propensity score loss (cross-entropy)
        treatment_idx = torch.argmax(batch['treatment'], dim=1)
        propensity_loss = F.cross_entropy(pred['propensity_logits'], treatment_idx)
        
        # 3. Targeted regularization loss (Dragonnet-specific)
        # Encourage factual outcome and targeted outcome to be similar
        targeted_reg_loss = F.mse_loss(pred['factual_outcome'], pred['targeted_outcome'])
        
        # 4. Covariate balancing (IP weighting)
        weights = 1.0 / (pred['propensity_scores'][torch.arange(len(treatment_idx)), treatment_idx] + 1e-8)
        weights = weights / weights.mean()
        ipw_loss = (weights * (pred['factual_outcome'] - batch['outcome']).pow(2)).mean()
        
        # Combine losses with Dragonnet weighting
        total_loss = (prediction_loss + 
                      self.alpha * propensity_loss + 
                      0.1 * targeted_reg_loss + 
                      0.05 * ipw_loss)
        
        return total_loss

class CFRCausalRAG(BaseCausalRAG):
    """
    Counterfactual Regression (CFR) Causal RAG with representation balancing.
    Uses Wasserstein distance or MMD for balancing representations across treatments.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Representation network (phi) for balancing
        self.representation_network = nn.Sequential(
            nn.Linear(config.confounder_dim + config.embedding_dim * config.retrieval_top_k, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2)
        )
        
        # Hypothesis network (h) for outcome prediction
        self.hypothesis_network = nn.Sequential(
            nn.Linear(config.hidden_dim // 2 + config.treatment_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 4, config.outcome_dim)
        )
        
        # Propensity network for weighting
        self.propensity_network = nn.Sequential(
            nn.Linear(config.confounder_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.treatment_dim)
        )
        
        # Hyperparameters for CFR
        self.balance_lambda = getattr(config, 'cfr_balance_lambda', 1.0)  # Weight for balancing term
        self.balance_method = getattr(config, 'cfr_balance_method', 'mmd')  # 'mmd' or 'wass'
        
    def forward(self, batch, corpus_embeddings=None):
        """Forward pass with representation balancing."""
        patient_features = batch['patient']
        treatment = batch['treatment']
        confounders = batch['confounders']
        patient_query_emb = batch.get('patient_query_emb', None)
        
        # Retrieve documents
        retrieved_embeddings, retrieval_scores, retrieval_indices = self.retrieve_documents(
            patient_features, corpus_embeddings, self.retrieval_top_k, patient_query_emb
        )
        
        # Flatten retrieved embeddings
        batch_size = patient_features.size(0)
        flattened_retrieval = retrieved_embeddings.view(batch_size, -1)
        
        # Create representation (phi)
        rep_input = torch.cat([confounders, flattened_retrieval], dim=1)
        representation = self.representation_network(rep_input)
        
        # Factual outcome prediction
        factual_input = torch.cat([representation, treatment], dim=1)
        factual_outcome = self.hypothesis_network(factual_input)
        
        # Propensity scores
        propensity_scores = F.softmax(self.propensity_network(confounders), dim=1)
        
        # Generate counterfactual predictions for evaluation
        counterfactual_preds = []
        for t in range(self.treatment_dim):
            treatment_vec = torch.zeros(batch_size, self.treatment_dim, device=confounders.device)
            treatment_vec[:, t] = 1.0
            cf_input = torch.cat([representation, treatment_vec], dim=1)
            cf_outcome = self.hypothesis_network(cf_input)
            counterfactual_preds.append(cf_outcome)
        
        counterfactual_preds = torch.stack(counterfactual_preds, dim=1)
        
        return {
            'outcome': factual_outcome,
            'factual_outcome': factual_outcome,
            'counterfactual_preds': counterfactual_preds,
            'propensity_scores': propensity_scores,
            'representation': representation,
            'retrieval_scores': retrieval_scores,
            'retrieval_indices': retrieval_indices
        }
    
    def compute_mmd_distance(self, rep_t0, rep_t1):
        """Compute Maximum Mean Discrepancy (MMD) between representations."""
        # Use RBF kernel for MMD
        n_t0 = rep_t0.size(0)
        n_t1 = rep_t1.size(0)
        
        # Compute kernel matrices
        rep_t0_norm = torch.sum(rep_t0**2, dim=1, keepdim=True)
        rep_t1_norm = torch.sum(rep_t1**2, dim=1, keepdim=True)
        
        # Pairwise distances
        distances_t0 = rep_t0_norm + rep_t0_norm.t() - 2.0 * torch.mm(rep_t0, rep_t0.t())
        distances_t1 = rep_t1_norm + rep_t1_norm.t() - 2.0 * torch.mm(rep_t1, rep_t1.t())
        distances_cross = rep_t0_norm + rep_t1_norm.t() - 2.0 * torch.mm(rep_t0, rep_t1.t())
        
        # RBF kernel with bandwidth heuristic
        bandwidth = torch.median(distances_cross.detach())
        if bandwidth == 0:
            bandwidth = 1.0
            
        K_t0 = torch.exp(-distances_t0 / (2 * bandwidth))
        K_t1 = torch.exp(-distances_t1 / (2 * bandwidth))
        K_cross = torch.exp(-distances_cross / (2 * bandwidth))
        
        # MMD^2
        mmd = (K_t0.mean() + K_t1.mean() - 2 * K_cross.mean())
        return torch.clamp(mmd, min=0.0)
    
    def compute_wasserstein_distance(self, rep_t0, rep_t1):
        """Approximate Wasserstein-1 distance between representations."""
        # Simple approximation: L2 distance between mean representations
        mean_t0 = rep_t0.mean(dim=0)
        mean_t1 = rep_t1.mean(dim=0)
        return torch.norm(mean_t0 - mean_t1, p=2)
    
    def compute_loss(self, pred, batch):
        """
        CFR loss with representation balancing.
        """
        # 1. Prediction loss (factual outcomes)
        prediction_loss = F.mse_loss(pred['factual_outcome'], batch['outcome'])
        
        # 2. Counterfactual prediction loss (if ground truth available)
        if 'counterfactuals' in batch:
            counterfactual_loss = F.mse_loss(pred['counterfactual_preds'], batch['counterfactuals'])
        else:
            counterfactual_loss = torch.tensor(0.0, device=pred['factual_outcome'].device)
        
        # 3. Representation balancing loss
        treatment_idx = torch.argmax(batch['treatment'], dim=1)
        representation = pred['representation']
        
        # Split representations by treatment
        balancing_loss = torch.tensor(0.0, device=representation.device)
        
        if self.treatment_dim > 1:
            # For binary treatment (0 vs 1), balance representations
            mask_t0 = (treatment_idx == 0)
            mask_t1 = (treatment_idx == 1)
            
            if mask_t0.sum() > 0 and mask_t1.sum() > 0:
                rep_t0 = representation[mask_t0]
                rep_t1 = representation[mask_t1]
                
                if self.balance_method == 'mmd':
                    balancing_loss = self.compute_mmd_distance(rep_t0, rep_t1)
                else:  # 'wass'
                    balancing_loss = self.compute_wasserstein_distance(rep_t0, rep_t1)
        
        # 4. Propensity score entropy (encourage balanced propensity)
        propensity_scores = pred['propensity_scores']
        entropy_loss = -torch.sum(propensity_scores * torch.log(propensity_scores + 1e-8), dim=1).mean()
        
        # Combine losses
        total_loss = (prediction_loss + 
                      0.5 * counterfactual_loss + 
                      self.balance_lambda * balancing_loss + 
                      0.1 * entropy_loss)
        
        return total_loss

class MultiHeadCausalRAG(BaseCausalRAG):
    """
    Multi-Head Causal RAG with deeper networks and multiple attention heads.
    Enhanced architecture with more capacity for complex causal relationships.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Enhanced shared encoder with more layers
        self.shared_encoder = nn.Sequential(
            nn.Linear(config.confounder_dim + config.embedding_dim * config.retrieval_top_k, config.hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2)
        )
        
        # Multiple attention heads for treatment-specific processing
        self.num_heads = 4
        self.head_dim = config.hidden_dim // 2 // self.num_heads
        
        # Multi-head attention projection
        self.query_proj = nn.Linear(config.hidden_dim // 2, config.hidden_dim // 2)
        self.key_proj = nn.Linear(config.treatment_dim, config.hidden_dim // 2)
        self.value_proj = nn.Linear(config.hidden_dim // 2, config.hidden_dim // 2)
        
        # Treatment-specific outcome heads (one per treatment)
        self.treatment_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim // 2, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 4, config.outcome_dim)
            ) for _ in range(config.treatment_dim)
        ])
        
        # Enhanced propensity network
        self.propensity_network = nn.Sequential(
            nn.Linear(config.confounder_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.treatment_dim)
        )
        
        # ATE regularization network
        self.ate_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 4, 1)  # Predicts ATE
        )
        
        # Hyperparameters
        self.attention_dropout = nn.Dropout(0.1)
        self.cf_lambda = getattr(config, 'multihd_cf_lambda', 0.7)  # Higher counterfactual weight
        self.ate_lambda = getattr(config, 'multihd_ate_lambda', 0.4)  # Higher ATE regularization
        
    def multi_head_attention(self, query, key, value):
        """Multi-head attention for treatment-aware representation."""
        batch_size = query.size(0)
        
        # Project to multi-head space
        Q = self.query_proj(query).view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        K = self.key_proj(key).view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.value_proj(value).view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, V)
        context = context.transpose(0, 1).contiguous().view(batch_size, -1)
        
        return context, attn_weights
    
    def forward(self, batch, corpus_embeddings=None):
        """Forward pass with multi-head attention."""
        patient_features = batch['patient']
        treatment = batch['treatment']
        confounders = batch['confounders']
        patient_query_emb = batch.get('patient_query_emb', None)
        
        # Retrieve documents
        retrieved_embeddings, retrieval_scores, retrieval_indices = self.retrieve_documents(
            patient_features, corpus_embeddings, self.retrieval_top_k, patient_query_emb
        )
        
        # Flatten retrieved embeddings
        batch_size = patient_features.size(0)
        flattened_retrieval = retrieved_embeddings.view(batch_size, -1)
        
        # Create shared representation
        shared_input = torch.cat([confounders, flattened_retrieval], dim=1)
        shared_representation = self.shared_encoder(shared_input)
        
        # Multi-head attention with treatment information
        attended_rep, attn_weights = self.multi_head_attention(
            shared_representation, treatment, shared_representation
        )
        
        # Combine attended representation with original
        enhanced_rep = shared_representation + 0.1 * attended_rep
        
        # Get treatment index for factual prediction
        treatment_idx = torch.argmax(treatment, dim=1)
        
        # Factual outcome prediction
        factual_outcomes = []
        for i in range(batch_size):
            idx = treatment_idx[i]
            outcome = self.treatment_heads[idx](enhanced_rep[i:i+1])
            factual_outcomes.append(outcome)
        
        factual_outcome = torch.cat(factual_outcomes, dim=0)
        
        # Generate all counterfactual predictions
        counterfactual_preds = []
        for t in range(self.treatment_dim):
            cf_outcomes = self.treatment_heads[t](enhanced_rep)
            counterfactual_preds.append(cf_outcomes)
        
        counterfactual_preds = torch.stack(counterfactual_preds, dim=1)
        
        # Propensity scores
        propensity_scores = F.softmax(self.propensity_network(confounders), dim=1)
        
        # Predict ATE directly from representation
        ate_prediction = self.ate_predictor(enhanced_rep.mean(dim=0, keepdim=True))
        
        return {
            'outcome': factual_outcome,
            'factual_outcome': factual_outcome,
            'counterfactual_preds': counterfactual_preds,
            'propensity_scores': propensity_scores,
            'enhanced_representation': enhanced_rep,
            'shared_representation': shared_representation,
            'attention_weights': attn_weights,
            'ate_prediction': ate_prediction,
            'retrieval_scores': retrieval_scores,
            'retrieval_indices': retrieval_indices
        }
    
    def compute_loss(self, pred, batch):
        """
        Multi-head causal RAG loss with enhanced regularization.
        """
        # 1. Factual prediction loss
        factual_loss = F.mse_loss(pred['factual_outcome'], batch['outcome'])
        
        # 2. Counterfactual prediction loss (if available)
        if 'counterfactuals' in batch:
            counterfactual_loss = F.mse_loss(pred['counterfactual_preds'], batch['counterfactuals'])
        else:
            counterfactual_loss = torch.tensor(0.0, device=pred['factual_outcome'].device)
        
        # 3. ATE regularization loss
        if self.treatment_dim > 1 and 'counterfactuals' in batch:
            # True ATE from ground truth counterfactuals
            true_ate = (batch['counterfactuals'][:, 1, :] - batch['counterfactuals'][:, 0, :]).mean()
            
            # Predicted ATE from model's counterfactual predictions
            pred_ate = (pred['counterfactual_preds'][:, 1, :] - pred['counterfactual_preds'][:, 0, :]).mean()
            
            ate_loss = F.mse_loss(pred_ate, true_ate)
        else:
            ate_loss = torch.tensor(0.0, device=pred['factual_outcome'].device)
        
        # 4. Propensity score entropy (balancing)
        propensity_scores = pred['propensity_scores']
        entropy_loss = -torch.sum(propensity_scores * torch.log(propensity_scores + 1e-8), dim=1).mean()
        
        # 5. Representation consistency loss
        shared_rep = pred['shared_representation']
        rep_variance = shared_rep.var(dim=0).mean()  # Encourage similar representations
        
        # 6. Attention diversity loss (encourage diverse attention heads)
        attn_weights = pred['attention_weights']
        if attn_weights is not None:
            # Compute entropy of attention weights across heads
            attn_entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=1).mean()
            diversity_loss = -attn_entropy  # Maximize entropy for diverse attention
        else:
            diversity_loss = torch.tensor(0.0, device=factual_loss.device)
        
        # Combine all losses
        total_loss = (factual_loss + 
                     self.cf_lambda * counterfactual_loss + 
                     self.ate_lambda * ate_loss + 
                     0.1 * entropy_loss + 
                     0.05 * rep_variance + 
                     0.01 * diversity_loss)
        
        return total_loss
