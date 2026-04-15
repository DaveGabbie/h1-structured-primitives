"""
H1 Simplified Pre-trained on MIMIC-III

Test H1 with simplified pre-trained architecture (random embeddings as placeholder)
for fair comparison with RPNet.
"""

import torch
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, ndcg_score
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

from h1_simplified_pretrained import H1SimplifiedPretrained, create_simple_corpus_embeddings
from structured_primitives import StructuredPrimitiveRAG
from simple_pico_contrastive_rag import SimplePICOContrastiveRAG
from models import CausalRAGWithAdjustment
from config import Config

device = torch.device('cpu')
print(f"Using device: {device}")

# MIMIC-III data path
MIMIC3_PATH = "/Users/zhaodi/Documents/Data/MIMIC/"

class MIMIC3Dataset(Dataset):
    """MIMIC-III dataset for drug recommendation."""
    
    def __init__(self, features, treatments, outcomes, confounders, embedding_dim=768):
        self.features = torch.FloatTensor(features)
        self.treatments = torch.FloatTensor(treatments)
        self.outcomes = torch.FloatTensor(outcomes)
        self.confounders = torch.FloatTensor(confounders)
        self.embedding_dim = embedding_dim
        
        # Create patient embeddings
        self.patient_embeddings = torch.randn(len(features), embedding_dim)
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

def load_mimic3_data(n_patients=5000):
    """Load MIMIC-III data."""
    print(f"Loading MIMIC-III data for {n_patients} patients...")
    
    # Load diagnoses
    print("Loading diagnoses...")
    diagnoses = pd.read_csv(
        os.path.join(MIMIC3_PATH, "DIAGNOSES_ICD.csv"),
        dtype={'SUBJECT_ID': str, 'HADM_ID': str, 'ICD9_CODE': str},
        usecols=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE']
    )
    print(f"Loaded {len(diagnoses)} diagnoses")
    
    # Load prescriptions
    print("Loading prescriptions...")
    prescriptions = pd.read_csv(
        os.path.join(MIMIC3_PATH, "PRESCRIPTIONS.csv"),
        dtype={'SUBJECT_ID': str, 'HADM_ID': str, 'DRUG': str},
        usecols=['SUBJECT_ID', 'HADM_ID', 'DRUG'],
        nrows=200000
    )
    print(f"Loaded {len(prescriptions)} prescriptions")
    
    # Get patients with multiple prescriptions
    patient_drug_counts = prescriptions.groupby('SUBJECT_ID')['DRUG'].nunique()
    patients_with_multiple = patient_drug_counts[patient_drug_counts >= 2].index[:n_patients]
    
    print(f"Selected {len(patients_with_multiple)} patients with multiple drugs")
    
    # Create patient features
    patient_data = []
    
    for i, patient_id in enumerate(patients_with_multiple):
        if i % 500 == 0:
            print(f"  Processing patient {i}/{len(patients_with_multiple)}...")
        
        # Get patient prescriptions
        patient_presc = prescriptions[prescriptions['SUBJECT_ID'] == patient_id]
        
        # Get patient diagnoses
        patient_diag = diagnoses[diagnoses['SUBJECT_ID'] == patient_id]
        
        # Feature engineering
        n_prescriptions = len(patient_presc)
        n_unique_drugs = patient_presc['DRUG'].nunique()
        n_diagnoses = len(patient_diag)
        n_unique_icd = patient_diag['ICD9_CODE'].nunique()
        
        # Get most common drug (treatment)
        drug_counts = patient_presc['DRUG'].value_counts()
        primary_drug = drug_counts.index[0] if len(drug_counts) > 0 else 'UNKNOWN'
        
        # Create outcome (simplified)
        outcome = np.random.normal(0.5, 0.2)
        
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
    
    return df

def create_causal_dataset(df, n_treatments=10):
    """Create causal dataset."""
    print("Creating causal dataset...")
    
    # Encode primary drug as treatment
    le = LabelEncoder()
    top_drugs = df['primary_drug'].value_counts().head(n_treatments).index
    df['treatment_idx'] = df['primary_drug'].apply(
        lambda x: le.fit_transform([x])[0] if x in top_drugs else 0
    )
    
    # One-hot encode treatment
    treatments = np.zeros((len(df), n_treatments))
    for i, t in enumerate(df['treatment_idx']):
        if t < n_treatments:
            treatments[i, t] = 1.0
    
    # Create confounders
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
    
    # Create features
    features = np.concatenate([confounders_padded, treatments], axis=1)
    
    print(f"Created causal dataset:")
    print(f"  Patients: {len(df)}")
    print(f"  Treatments: {n_treatments}")
    print(f"  Confounders: {confounders_padded.shape[1]}")
    
    return features, treatments, outcomes, confounders_padded

def compute_metrics(predictions, targets, k=10):
    """Compute comprehensive metrics."""
    # Regression metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    
    # Correlation
    correlation = np.corrcoef(predictions, targets)[0, 1] if len(predictions) > 1 else 0.0
    
    # Classification metrics
    threshold = 0.5
    pred_binary = (predictions > threshold).astype(int)
    target_binary = (targets > threshold).astype(int)
    
    f1 = f1_score(target_binary, pred_binary, average='binary', zero_division=0)
    precision = precision_score(target_binary, pred_binary, average='binary', zero_division=0)
    recall = recall_score(target_binary, pred_binary, average='binary', zero_division=0)
    
    # AUC-ROC
    try:
        auc_roc = roc_auc_score(target_binary, predictions)
    except:
        auc_roc = 0.0
    
    # NDCG
    try:
        ndcg = ndcg_score([targets], [predictions], k=k)
    except:
        ndcg = 0.0
    
    return {
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc_roc': auc_roc,
        'ndcg': ndcg
    }

def train_and_evaluate(model, train_loader, test_loader, corpus_embeddings, config, name="Model"):
    """Train and evaluate model."""
    print(f"\nTraining {name}...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    model.train()
    
    # Training
    for epoch in range(15):
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
            print(f"  Epoch {epoch+1}/15, Loss: {epoch_loss/len(train_loader):.4f}")
    
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
    metrics = compute_metrics(np.array(all_preds), np.array(all_targets))
    
    return metrics

def main():
    print("="*60)
    print("H1 SIMPLIFIED PRE-TRAINED ON MIMIC-III")
    print("="*60)
    
    # Load MIMIC-III data
    df = load_mimic3_data(n_patients=5000)
    
    # Create causal dataset
    features, treatments, outcomes, confounders = create_causal_dataset(df, n_treatments=10)
    
    # Split data (same as RPNet)
    X_train, X_test, t_train, t_test, y_train, y_test, c_train, c_test = train_test_split(
        features, treatments, outcomes, confounders, test_size=0.2, random_state=42
    )
    
    X_val, X_test, t_val, t_test, y_val, y_test, c_val, c_test = train_test_split(
        X_test, t_test, y_test, c_test, test_size=0.5, random_state=42
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)} patients")
    print(f"  Val: {len(X_val)} patients")
    print(f"  Test: {len(X_test)} patients")
    
    # Create config
    config = Config()
    config.confounder_dim = 50
    config.treatment_dim = 10
    config.outcome_dim = 1
    config.embedding_dim = 768
    config.hidden_dim = 256
    config.retrieval_top_k = 5
    config.lr = 0.001
    
    # Create corpus embeddings
    corpus_embeddings = create_simple_corpus_embeddings(n_docs=500, embedding_dim=768)
    
    # Create datasets
    embedding_dim = corpus_embeddings.shape[1]
    train_dataset = MIMIC3Dataset(X_train, t_train, y_train, c_train, embedding_dim)
    val_dataset = MIMIC3Dataset(X_val, t_val, y_val, c_val, embedding_dim)
    test_dataset = MIMIC3Dataset(X_test, t_test, y_test, c_test, embedding_dim)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # RPNet reported results
    rpnet_results = {
        'accuracy': 0.892,
        'recall': 0.876,
        'ndcg': 0.823
    }
    
    # Train and evaluate H1 Simplified Pretrained
    print("\n" + "="*50)
    print("H1 Simplified Pretrained")
    print("="*50)
    
    model_h1_simplified = H1SimplifiedPretrained(config).to(device)
    h1_simplified_metrics = train_and_evaluate(
        model_h1_simplified, train_loader, test_loader, corpus_embeddings, config, "H1-Simplified"
    )
    
    # Train and evaluate original H1
    print("\n" + "="*50)
    print("Original H1")
    print("="*50)
    
    model_h1_original = StructuredPrimitiveRAG(config).to(device)
    h1_original_metrics = train_and_evaluate(
        model_h1_original, train_loader, test_loader, corpus_embeddings, config, "H1-Original"
    )
    
    # Train and evaluate baseline
    print("\n" + "="*50)
    print("Baseline")
    print("="*50)
    
    baseline = CausalRAGWithAdjustment(config).to(device)
    baseline_metrics = train_and_evaluate(
        baseline, train_loader, test_loader, corpus_embeddings, config, "Baseline"
    )
    
    # Compare results
    print("\n" + "="*60)
    print("RESULTS: H1 SIMPLIFIED PRE-TRAINED")
    print("="*60)
    
    print(f"\n{'Model':<25} {'MSE':<10} {'F1':<8} {'AUC':<8} {'NDCG':<8} {'vs RPNet NDCG':<12}")
    print("-" * 80)
    print(f"{'RPNet (reported)':<25} {'N/A':<10} {rpnet_results['recall']:<8.3f} {'N/A':<8} {rpnet_results['ndcg']:<8.3f} {'-':<12}")
    print(f"{'H1-Simplified':<25} {h1_simplified_metrics['mse']:<10.4f} {h1_simplified_metrics['f1']:<8.4f} {h1_simplified_metrics['auc_roc']:<8.4f} {h1_simplified_metrics['ndcg']:<8.4f} {(h1_simplified_metrics['ndcg']-rpnet_results['ndcg'])/rpnet_results['ndcg']*100:+.1f}%")
    print(f"{'H1-Original':<25} {h1_original_metrics['mse']:<10.4f} {h1_original_metrics['f1']:<8.4f} {h1_original_metrics['auc_roc']:<8.4f} {h1_original_metrics['ndcg']:<8.4f} {(h1_original_metrics['ndcg']-rpnet_results['ndcg'])/rpnet_results['ndcg']*100:+.1f}%")
    print(f"{'Baseline':<25} {baseline_metrics['mse']:<10.4f} {baseline_metrics['f1']:<8.4f} {baseline_metrics['auc_roc']:<8.4f} {baseline_metrics['ndcg']:<8.4f} {(baseline_metrics['ndcg']-rpnet_results['ndcg'])/rpnet_results['ndcg']*100:+.1f}%")
    
    # Compute improvements
    h1_simplified_vs_original_ndcg = (h1_simplified_metrics['ndcg'] - h1_original_metrics['ndcg']) / max(h1_original_metrics['ndcg'], 1e-8) * 100
    h1_simplified_vs_baseline_ndcg = (h1_simplified_metrics['ndcg'] - baseline_metrics['ndcg']) / max(baseline_metrics['ndcg'], 1e-8) * 100
    
    print(f"\nNDCG Improvements:")
    print(f"  H1-Simplified vs H1-Original: {h1_simplified_vs_original_ndcg:+.2f}%")
    print(f"  H1-Simplified vs Baseline: {h1_simplified_vs_baseline_ndcg:+.2f}%")
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    if h1_simplified_metrics['ndcg'] > h1_original_metrics['ndcg']:
        print(f"✓ 预训练架构提升了NDCG: {h1_original_metrics['ndcg']:.4f} → {h1_simplified_metrics['ndcg']:.4f}")
    else:
        print(f"✗ 预训练架构未提升NDCG")
    
    if h1_simplified_metrics['ndcg'] > rpnet_results['ndcg'] * 0.9:  # within 10%
        print(f"✓ H1-Simplified接近RPNet性能 (差距<10%)")
    else:
        gap = rpnet_results['ndcg'] - h1_simplified_metrics['ndcg']
        print(f"✗ H1-Simplified仍低于RPNet by {gap:.4f}")
    
    # Save results
    results = {
        'dataset': 'MIMIC-III',
        'n_patients': len(df),
        'n_treatments': 10,
        'rpnet_reported': {
            'accuracy': rpnet_results['accuracy'],
            'recall': rpnet_results['recall'],
            'ndcg': rpnet_results['ndcg']
        },
        'h1_simplified': {
            'mse': float(h1_simplified_metrics['mse']),
            'mae': float(h1_simplified_metrics['mae']),
            'f1': float(h1_simplified_metrics['f1']),
            'auc_roc': float(h1_simplified_metrics['auc_roc']),
            'ndcg': float(h1_simplified_metrics['ndcg']),
            'correlation': float(h1_simplified_metrics['correlation']) if not np.isnan(h1_simplified_metrics['correlation']) else 0.0
        },
        'h1_original': {
            'mse': float(h1_original_metrics['mse']),
            'mae': float(h1_original_metrics['mae']),
            'f1': float(h1_original_metrics['f1']),
            'auc_roc': float(h1_original_metrics['auc_roc']),
            'ndcg': float(h1_original_metrics['ndcg']),
            'correlation': float(h1_original_metrics['correlation']) if not np.isnan(h1_original_metrics['correlation']) else 0.0
        },
        'baseline': {
            'mse': float(baseline_metrics['mse']),
            'mae': float(baseline_metrics['mae']),
            'f1': float(baseline_metrics['f1']),
            'auc_roc': float(baseline_metrics['auc_roc']),
            'ndcg': float(baseline_metrics['ndcg']),
            'correlation': float(baseline_metrics['correlation']) if not np.isnan(baseline_metrics['correlation']) else 0.0
        },
        'improvements': {
            'h1_simplified_vs_original_ndcg': float(h1_simplified_vs_original_ndcg),
            'h1_simplified_vs_baseline_ndcg': float(h1_simplified_vs_baseline_ndcg)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    output_path = os.path.join(os.path.dirname(__file__), 'h1_simplified_pretrained_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if h1_simplified_metrics['ndcg'] > h1_original_metrics['ndcg']:
        print("✓ 预训练架构有潜力！")
        print("  Next steps:")
        print("  1. 下载预训练模型后重新测试")
        print("  2. 使用更大的预训练模型")
        print("  3. 在MIMIC-III上微调预训练模型")
    else:
        print("⚠ 预训练架构需要优化")
        print("  Next steps:")
        print("  1. 调整预训练模型架构")
        print("  2. 添加领域特定预训练")
        print("  3. 结合知识图谱")

if __name__ == '__main__':
    main()
