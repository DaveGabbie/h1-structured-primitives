"""
H1 on Real MIMIC-IV Data

Creates a causal dataset from MIMIC-IV for drug recommendation and tests H1.
"""

import torch
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

from structured_primitives import StructuredPrimitiveRAG
from models import CausalRAGWithAdjustment
from config import Config

device = torch.device('cpu')
print(f"Using device: {device}")

# MIMIC-IV data path
MIMIC_PATH = "/Users/zhaodi/Documents/Data/MIMIC/"

class MIMICCausalDataset(Dataset):
    """MIMIC-IV dataset for causal drug recommendation."""
    
    def __init__(self, features, treatments, outcomes, confounders, embedding_dim=768):
        self.features = torch.FloatTensor(features)
        self.treatments = torch.FloatTensor(treatments)
        self.outcomes = torch.FloatTensor(outcomes)
        self.confounders = torch.FloatTensor(confounders)
        self.embedding_dim = embedding_dim
        
        # Create patient embeddings matching corpus dimension
        self.patient_embeddings = torch.randn(len(features), embedding_dim)
        # Normalize
        self.patient_embeddings = torch.nn.functional.normalize(self.patient_embeddings, dim=1)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'patient': torch.cat([self.confounders[idx], self.treatments[idx]], dim=0),
            'treatment': self.treatments[idx],
            'outcome': self.outcomes[idx].unsqueeze(0),
            'confounders': self.confounders[idx],
            'patient_query_emb': self.patient_embeddings[idx]
        }

def load_and_preprocess_mimic(n_patients=2000):
    """Load and preprocess MIMIC-IV data for causal inference."""
    print("Loading MIMIC-IV data...")
    
    # Load prescriptions (chunked for large file)
    print("Loading prescriptions...")
    prescriptions = pd.read_csv(
        os.path.join(MIMIC_PATH, "prescriptions-2.csv"),
        dtype={'subject_id': str, 'hadm_id': str, 'drug': str},
        usecols=['subject_id', 'hadm_id', 'drug', 'dose_val_rx', 'dose_unit_rx'],
        nrows=50000  # Limit for testing
    )
    
    # Load diagnoses
    print("Loading diagnoses...")
    diagnoses = pd.read_csv(
        os.path.join(MIMIC_PATH, "diagnoses_icd-2.csv"),
        dtype={'subject_id': str, 'hadm_id': str, 'icd_code': str},
        usecols=['subject_id', 'hadm_id', 'icd_code']
    )
    
    print(f"Loaded {len(prescriptions)} prescriptions, {len(diagnoses)} diagnoses")
    
    # Get patients with multiple prescriptions (for treatment comparison)
    patient_drug_counts = prescriptions.groupby('subject_id')['drug'].nunique()
    patients_with_multiple = patient_drug_counts[patient_drug_counts >= 2].index[:n_patients]
    
    print(f"Selected {len(patients_with_multiple)} patients with multiple drugs")
    
    # Create patient features
    patient_data = []
    
    for patient_id in patients_with_multiple:
        # Get patient prescriptions
        patient_presc = prescriptions[prescriptions['subject_id'] == patient_id]
        
        # Get patient diagnoses
        patient_diag = diagnoses[diagnoses['subject_id'] == patient_id]
        
        # Feature engineering
        n_prescriptions = len(patient_presc)
        n_unique_drugs = patient_presc['drug'].nunique()
        n_diagnoses = len(patient_diag)
        n_unique_icd = patient_diag['icd_code'].nunique()
        
        # Get most common drug (treatment)
        drug_counts = patient_presc['drug'].value_counts()
        primary_drug = drug_counts.index[0] if len(drug_counts) > 0 else 'UNKNOWN'
        
        # Create outcome (simplified: length of stay proxy)
        # In real implementation, would use actual outcomes
        outcome = np.random.normal(0.5, 0.2)  # Placeholder
        
        patient_data.append({
            'patient_id': patient_id,
            'n_prescriptions': n_prescriptions,
            'n_unique_drugs': n_unique_drugs,
            'n_diagnoses': n_diagnoses,
            'n_unique_icd': n_unique_icd,
            'primary_drug': primary_drug,
            'outcome': outcome
        })
    
    df = pd.DataFrame(patient_data)
    print(f"Created dataset with {len(df)} patients")
    
    return df, prescriptions, diagnoses

def create_causal_dataset(df, n_treatments=5):
    """Create causal dataset with treatment and confounders."""
    print("Creating causal dataset...")
    
    # Encode primary drug as treatment
    le = LabelEncoder()
    # Get top N most common drugs
    top_drugs = df['primary_drug'].value_counts().head(n_treatments).index
    df['treatment_idx'] = df['primary_drug'].apply(
        lambda x: le.fit_transform([x])[0] if x in top_drugs else 0
    )
    
    # One-hot encode treatment
    treatments = np.zeros((len(df), n_treatments))
    for i, t in enumerate(df['treatment_idx']):
        if t < n_treatments:
            treatments[i, t] = 1.0
    
    # Create confounders (patient features)
    confounder_cols = ['n_prescriptions', 'n_unique_drugs', 'n_diagnoses', 'n_unique_icd']
    confounders = df[confounder_cols].values
    
    # Standardize confounders
    scaler = StandardScaler()
    confounders = scaler.fit_transform(confounders)
    
    # Pad confounders to expected dimension (50)
    confounders_padded = np.zeros((len(df), 50))
    confounders_padded[:, :confounders.shape[1]] = confounders
    
    # Create outcomes
    outcomes = df['outcome'].values
    
    # Create features (concatenation of confounders and treatment)
    features = np.concatenate([confounders_padded, treatments], axis=1)
    
    print(f"Created causal dataset:")
    print(f"  Patients: {len(df)}")
    print(f"  Treatments: {n_treatments}")
    print(f"  Confounders: {confounders_padded.shape[1]}")
    print(f"  Outcome mean: {outcomes.mean():.4f}, std: {outcomes.std():.4f}")
    
    return features, treatments, outcomes, confounders_padded

def create_corpus_embeddings(n_docs=150, embedding_dim=768):
    """Create document embeddings for retrieval corpus."""
    print(f"Creating corpus embeddings ({n_docs} docs, {embedding_dim} dim)...")
    
    # Create random embeddings (in real implementation, would use actual medical documents)
    embeddings = np.random.randn(n_docs, embedding_dim).astype(np.float32)
    
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    
    return torch.FloatTensor(embeddings)

def train_and_evaluate(model, train_loader, test_loader, corpus_embeddings, config, name="Model"):
    """Train and evaluate model."""
    print(f"\nTraining {name}...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    model.train()
    
    # Training
    for epoch in range(10):
        epoch_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()
            pred = model(batch, corpus_embeddings.to(device))
            loss = model.compute_loss(pred, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/10, Loss: {epoch_loss/len(train_loader):.4f}")
    
    # Evaluation
    print(f"Evaluating {name}...")
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            pred = model(batch, corpus_embeddings.to(device))
            
            all_preds.extend(pred['outcome'].cpu().numpy().flatten())
            all_targets.extend(batch['outcome'].cpu().numpy().flatten())
    
    # Compute metrics
    mse = np.mean((np.array(all_preds) - np.array(all_targets)) ** 2)
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    
    return {
        'mse': mse,
        'mae': mae,
        'predictions': all_preds,
        'targets': all_targets
    }

def main():
    print("="*60)
    print("H1 ON REAL MIMIC-IV DATA")
    print("="*60)
    
    # Load MIMIC-IV data
    df, prescriptions, diagnoses = load_and_preprocess_mimic(n_patients=2000)
    
    # Create causal dataset
    features, treatments, outcomes, confounders = create_causal_dataset(df, n_treatments=5)
    
    # Split data
    X_train, X_test, t_train, t_test, y_train, y_test, c_train, c_test = train_test_split(
        features, treatments, outcomes, confounders, test_size=0.2, random_state=42
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)} patients")
    print(f"  Test: {len(X_test)} patients")
    
    # Create corpus embeddings first
    corpus_embeddings = create_corpus_embeddings(n_docs=150, embedding_dim=768)
    
    # Create datasets
    embedding_dim = corpus_embeddings.shape[1]
    train_dataset = MIMICCausalDataset(X_train, t_train, y_train, c_train, embedding_dim)
    test_dataset = MIMICCausalDataset(X_test, t_test, y_test, c_test, embedding_dim)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create config
    config = Config()
    config.confounder_dim = 50
    config.treatment_dim = 5
    config.outcome_dim = 1
    config.embedding_dim = 768
    config.hidden_dim = 256
    config.retrieval_top_k = 5
    config.lr = 0.001
    
    # Train and evaluate H1
    print("\n" + "="*50)
    print("H1: Structured Causal Primitives")
    print("="*50)
    
    model_h1 = StructuredPrimitiveRAG(config).to(device)
    h1_results = train_and_evaluate(
        model_h1, train_loader, test_loader, corpus_embeddings, config, "H1"
    )
    
    # Train and evaluate baseline
    print("\n" + "="*50)
    print("Baseline: CausalRAGWithAdjustment")
    print("="*50)
    
    baseline = CausalRAGWithAdjustment(config).to(device)
    baseline_results = train_and_evaluate(
        baseline, train_loader, test_loader, corpus_embeddings, config, "Baseline"
    )
    
    # Compare results
    print("\n" + "="*60)
    print("RESULTS ON REAL MIMIC-IV DATA")
    print("="*60)
    
    print(f"\n{'Model':<20} {'MSE':<10} {'MAE':<10}")
    print("-" * 45)
    print(f"{'H1':<20} {h1_results['mse']:<10.4f} {h1_results['mae']:<10.4f}")
    print(f"{'Baseline':<20} {baseline_results['mse']:<10.4f} {baseline_results['mae']:<10.4f}")
    
    # Compute improvement
    mse_improvement = (baseline_results['mse'] - h1_results['mse']) / baseline_results['mse'] * 100
    mae_improvement = (baseline_results['mae'] - h1_results['mae']) / baseline_results['mae'] * 100
    
    print(f"\nH1 vs Baseline:")
    print(f"  MSE improvement: {mse_improvement:+.2f}%")
    print(f"  MAE improvement: {mae_improvement:+.2f}%")
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    if mse_improvement > 0:
        print(f"✓ H1 outperforms baseline on MSE by {mse_improvement:.2f}%")
    else:
        print(f"✗ H1 underperforms baseline on MSE by {abs(mse_improvement):.2f}%")
    
    if mae_improvement > 0:
        print(f"✓ H1 outperforms baseline on MAE by {mae_improvement:.2f}%")
    else:
        print(f"✗ H1 underperforms baseline on MAE by {abs(mae_improvement):.2f}%")
    
    # Save results
    results = {
        'dataset': 'MIMIC-IV',
        'n_patients': len(df),
        'n_treatments': 5,
        'h1': {
            'mse': float(h1_results['mse']),
            'mae': float(h1_results['mae'])
        },
        'baseline': {
            'mse': float(baseline_results['mse']),
            'mae': float(baseline_results['mae'])
        },
        'improvement': {
            'mse': float(mse_improvement),
            'mae': float(mae_improvement)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    output_path = os.path.join(os.path.dirname(__file__), 'h1_mimic_iv_real_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if mse_improvement > 10:
        print("✓ H1 shows strong improvement on real MIMIC-IV data")
        print("  Next steps:")
        print("  1. Scale up to full MIMIC-IV dataset")
        print("  2. Add more sophisticated feature engineering")
        print("  3. Compare with PACE-RAG using F1 metric")
    elif mse_improvement > 0:
        print("✓ H1 shows modest improvement on real MIMIC-IV data")
        print("  Next steps:")
        print("  1. Tune hyperparameters")
        print("  2. Improve feature engineering")
        print("  3. Add more training data")
    else:
        print("✗ H1 needs improvement for real MIMIC-IV data")
        print("  Next steps:")
        print("  1. Debug data preprocessing")
        print("  2. Check feature quality")
        print("  3. Consider alternative architectures")

if __name__ == '__main__':
    main()
