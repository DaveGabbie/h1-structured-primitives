"""
Add Real Outcome Variables to MIMIC-IV

Load real clinical outcomes from MIMIC-IV data:
- Mortality (in-hospital death)
- Readmission (30-day readmission)
- Length of stay
- ICU admission
- Ventilator use
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
from simple_pico_contrastive_rag import SimplePICOContrastiveRAG
from models import CausalRAGWithAdjustment
from config import Config

device = torch.device('cpu')
print(f"Using device: {device}")

# MIMIC-IV data path
MIMIC_PATH = "/Users/zhaodi/Documents/Data/MIMIC/"

class MIMICCausalDataset(Dataset):
    """MIMIC-IV dataset for causal drug recommendation with real outcomes."""
    
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

def load_mimic_with_real_outcomes(n_patients=5000):
    """Load MIMIC-IV data with real clinical outcomes."""
    print(f"Loading MIMIC-IV data with real outcomes for {n_patients} patients...")
    
    # Load admissions (for mortality and readmission)
    print("Loading admissions...")
    try:
        admissions = pd.read_csv(
            os.path.join(MIMIC_PATH, "ADMISSIONS.csv"),
            dtype={'SUBJECT_ID': str, 'HADM_ID': str},
            usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ADMISSION_TYPE', 'ADMISSION_LOCATION']
        )
        print(f"Loaded {len(admissions)} admissions")
    except Exception as e:
        print(f"Error loading admissions: {e}")
        admissions = None
    
    # Load prescriptions
    print("Loading prescriptions...")
    prescriptions = pd.read_csv(
        os.path.join(MIMIC_PATH, "prescriptions-2.csv"),
        dtype={'subject_id': str, 'hadm_id': str, 'drug': str},
        usecols=['subject_id', 'hadm_id', 'drug'],
        nrows=100000
    )
    
    # Load diagnoses
    print("Loading diagnoses...")
    diagnoses = pd.read_csv(
        os.path.join(MIMIC_PATH, "diagnoses_icd-2.csv"),
        dtype={'subject_id': str, 'hadm_id': str, 'icd_code': str},
        usecols=['subject_id', 'hadm_id', 'icd_code']
    )
    
    print(f"Loaded {len(prescriptions)} prescriptions, {len(diagnoses)} diagnoses")
    
    # Get patients with multiple prescriptions
    patient_drug_counts = prescriptions.groupby('subject_id')['drug'].nunique()
    patients_with_multiple = patient_drug_counts[patient_drug_counts >= 2].index[:n_patients]
    
    print(f"Selected {len(patients_with_multiple)} patients with multiple drugs")
    
    # Create patient features with real outcomes
    patient_data = []
    
    for i, patient_id in enumerate(patients_with_multiple):
        if i % 500 == 0:
            print(f"  Processing patient {i}/{len(patients_with_multiple)}...")
        
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
        
        # Real outcomes from admissions
        mortality = 0.0
        readmission_30d = 0.0
        length_of_stay = 0.0
        
        if admissions is not None:
            # Get patient admissions
            patient_adm = admissions[admissions['SUBJECT_ID'] == patient_id]
            
            if len(patient_adm) > 0:
                # Mortality: check if patient died
                if patient_adm['DEATHTIME'].notna().any():
                    mortality = 1.0
                
                # Readmission: check if patient was readmitted within 30 days
                if len(patient_adm) > 1:
                    # Sort by admission time
                    patient_adm = patient_adm.sort_values('ADMITTIME')
                    # Check if any readmission within 30 days
                    for j in range(len(patient_adm) - 1):
                        admit_time = pd.to_datetime(patient_adm.iloc[j]['DISCHTIME'])
                        next_admit = pd.to_datetime(patient_adm.iloc[j + 1]['ADMITTIME'])
                        days_diff = (next_admit - admit_time).days
                        if 0 < days_diff <= 30:
                            readmission_30d = 1.0
                            break
                
                # Length of stay: average length of stay
                try:
                    patient_adm['ADMITTIME'] = pd.to_datetime(patient_adm['ADMITTIME'])
                    patient_adm['DISCHTIME'] = pd.to_datetime(patient_adm['DISCHTIME'])
                    los = (patient_adm['DISCHTIME'] - patient_adm['ADMITTIME']).dt.days.mean()
                    length_of_stay = los if not np.isnan(los) else 0.0
                except:
                    length_of_stay = 0.0
        
        # Create combined outcome (mortality + readmission + length of stay)
        # Weight: mortality (0.5), readmission (0.3), length of stay (0.2)
        outcome = 0.5 * mortality + 0.3 * readmission_30d + 0.2 * min(length_of_stay / 30, 1.0)
        
        patient_data.append({
            'patient_id': patient_id,
            'n_prescriptions': n_prescriptions,
            'n_unique_drugs': n_unique_drugs,
            'n_diagnoses': n_diagnoses,
            'n_unique_icd': n_unique_icd,
            'primary_drug': primary_drug,
            'mortality': mortality,
            'readmission_30d': readmission_30d,
            'length_of_stay': length_of_stay,
            'outcome': outcome
        })
    
    df = pd.DataFrame(patient_data)
    print(f"Created dataset with {len(df)} patients")
    
    # Print outcome statistics
    print(f"\nOutcome Statistics:")
    print(f"  Mortality rate: {df['mortality'].mean():.3f}")
    print(f"  30-day readmission rate: {df['readmission_30d'].mean():.3f}")
    print(f"  Average length of stay: {df['length_of_stay'].mean():.1f} days")
    print(f"  Combined outcome mean: {df['outcome'].mean():.3f}")
    
    return df

def create_causal_dataset(df, n_treatments=10):
    """Create causal dataset with real outcomes."""
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
    
    # Create outcomes (real clinical outcomes)
    outcomes = df['outcome'].values
    
    # Create features
    features = np.concatenate([confounders_padded, treatments], axis=1)
    
    print(f"Created causal dataset:")
    print(f"  Patients: {len(df)}")
    print(f"  Treatments: {n_treatments}")
    print(f"  Confounders: {confounders_padded.shape[1]}")
    print(f"  Outcome mean: {outcomes.mean():.4f}, std: {outcomes.std():.4f}")
    
    return features, treatments, outcomes, confounders_padded

def create_corpus_embeddings(n_docs=500, embedding_dim=768):
    """Create document embeddings."""
    print(f"Creating corpus embeddings ({n_docs} docs, {embedding_dim} dim)...")
    
    embeddings = np.random.randn(n_docs, embedding_dim).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    
    return torch.FloatTensor(embeddings)

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
    mse = np.mean((np.array(all_preds) - np.array(all_targets)) ** 2)
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    
    # Compute correlation
    correlation = np.corrcoef(all_preds, all_targets)[0, 1]
    
    return {
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'predictions': all_preds,
        'targets': all_targets
    }

def main():
    print("="*60)
    print("H1 & PICO-CONTRASTIVE RAG WITH REAL OUTCOMES")
    print("="*60)
    
    # Load MIMIC-IV data with real outcomes
    df = load_mimic_with_real_outcomes(n_patients=3000)
    
    # Create causal dataset
    features, treatments, outcomes, confounders = create_causal_dataset(df, n_treatments=10)
    
    # Split data
    X_train, X_test, t_train, t_test, y_train, y_test, c_train, c_test = train_test_split(
        features, treatments, outcomes, confounders, test_size=0.2, random_state=42
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)} patients")
    print(f"  Test: {len(X_test)} patients")
    
    # Create corpus embeddings
    corpus_embeddings = create_corpus_embeddings(n_docs=500, embedding_dim=768)
    
    # Create datasets
    embedding_dim = corpus_embeddings.shape[1]
    train_dataset = MIMICCausalDataset(X_train, t_train, y_train, c_train, embedding_dim)
    test_dataset = MIMICCausalDataset(X_test, t_test, y_test, c_test, embedding_dim)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Create config
    config = Config()
    config.confounder_dim = 50
    config.treatment_dim = 10
    config.outcome_dim = 1
    config.embedding_dim = 768
    config.hidden_dim = 256
    config.retrieval_top_k = 5
    config.lr = 0.001
    
    # Train and evaluate H1
    print("\n" + "="*50)
    print("H1: Structured Causal Primitives (Real Outcomes)")
    print("="*50)
    
    model_h1 = StructuredPrimitiveRAG(config).to(device)
    h1_results = train_and_evaluate(
        model_h1, train_loader, test_loader, corpus_embeddings, config, "H1"
    )
    
    # Train and evaluate PICO-Contrastive RAG
    print("\n" + "="*50)
    print("PICO-Contrastive RAG (Real Outcomes)")
    print("="*50)
    
    pico_rag = SimplePICOContrastiveRAG(config).to(device)
    pico_results = train_and_evaluate(
        pico_rag, train_loader, test_loader, corpus_embeddings, config, "PICO-Contrastive RAG"
    )
    
    # Train and evaluate baseline
    print("\n" + "="*50)
    print("Baseline: CausalRAGWithAdjustment (Real Outcomes)")
    print("="*50)
    
    baseline = CausalRAGWithAdjustment(config).to(device)
    baseline_results = train_and_evaluate(
        baseline, train_loader, test_loader, corpus_embeddings, config, "Baseline"
    )
    
    # Compare results
    print("\n" + "="*60)
    print("RESULTS WITH REAL OUTCOMES")
    print("="*60)
    
    print(f"\n{'Model':<25} {'MSE':<10} {'MAE':<10} {'Correlation':<12}")
    print("-" * 60)
    print(f"{'H1 (Ours)':<25} {h1_results['mse']:<10.4f} {h1_results['mae']:<10.4f} {h1_results['correlation']:<12.4f}")
    print(f"{'PICO-Contrastive RAG':<25} {pico_results['mse']:<10.4f} {pico_results['mae']:<10.4f} {pico_results['correlation']:<12.4f}")
    print(f"{'Baseline':<25} {baseline_results['mse']:<10.4f} {baseline_results['mae']:<10.4f} {baseline_results['correlation']:<12.4f}")
    
    # Compute improvements
    h1_vs_baseline_mse = (baseline_results['mse'] - h1_results['mse']) / baseline_results['mse'] * 100
    pico_vs_baseline_mse = (baseline_results['mse'] - pico_results['mse']) / baseline_results['mse'] * 100
    
    print(f"\nMSE Improvements vs Baseline:")
    print(f"  H1: {h1_vs_baseline_mse:+.2f}%")
    print(f"  PICO-Contrastive RAG: {pico_vs_baseline_mse:+.2f}%")
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    # Determine best model
    models = {
        'H1': h1_results['mse'],
        'PICO-Contrastive RAG': pico_results['mse'],
        'Baseline': baseline_results['mse']
    }
    best_model = min(models, key=models.get)  # Lower MSE is better
    best_mse = models[best_model]
    
    print(f"Best Model (lowest MSE): {best_model} (MSE: {best_mse:.4f})")
    
    if h1_results['correlation'] > 0.5:
        print(f"✓ H1 shows good correlation with real outcomes: {h1_results['correlation']:.4f}")
    else:
        print(f"⚠ H1 correlation with real outcomes is low: {h1_results['correlation']:.4f}")
    
    if pico_results['correlation'] > 0.5:
        print(f"✓ PICO-Contrastive RAG shows good correlation with real outcomes: {pico_results['correlation']:.4f}")
    else:
        print(f"⚠ PICO-Contrastive RAG correlation with real outcomes is low: {pico_results['correlation']:.4f}")
    
    # Save results
    results = {
        'dataset': 'MIMIC-IV with Real Outcomes',
        'n_patients': len(df),
        'n_treatments': 10,
        'outcome_statistics': {
            'mortality_rate': float(df['mortality'].mean()),
            'readmission_rate': float(df['readmission_30d'].mean()),
            'avg_length_of_stay': float(df['length_of_stay'].mean()),
            'combined_outcome_mean': float(df['outcome'].mean())
        },
        'h1': {
            'mse': float(h1_results['mse']),
            'mae': float(h1_results['mae']),
            'correlation': float(h1_results['correlation'])
        },
        'pico_contrastive_rag': {
            'mse': float(pico_results['mse']),
            'mae': float(pico_results['mae']),
            'correlation': float(pico_results['correlation'])
        },
        'baseline': {
            'mse': float(baseline_results['mse']),
            'mae': float(baseline_results['mae']),
            'correlation': float(baseline_results['correlation'])
        },
        'improvements': {
            'h1_vs_baseline_mse': float(h1_vs_baseline_mse),
            'pico_vs_baseline_mse': float(pico_vs_baseline_mse)
        },
        'best_model': {
            'name': best_model,
            'mse': float(best_mse)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    output_path = os.path.join(os.path.dirname(__file__), 'real_outcomes_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if best_mse < 0.1:
        print("✓ Excellent! Model achieves low MSE on real outcomes")
        print("  Next steps:")
        print("  1. Deploy in clinical setting")
        print("  2. Validate on external dataset")
        print("  3. Publish results")
    elif best_mse < 0.2:
        print("✓ Good! Model shows reasonable performance on real outcomes")
        print("  Next steps:")
        print("  1. Optimize hyperparameters")
        print("  2. Add more features")
        print("  3. Test on more patients")
    else:
        print("⚠ Model needs improvement for real outcomes")
        print("  Next steps:")
        print("  1. Debug feature engineering")
        print("  2. Try different outcome definitions")
        print("  3. Add more training data")

if __name__ == '__main__':
    main()
