"""
Compare H1 with PACE-RAG using F1 Score

PACE-RAG Benchmark:
- PACE-RAG F1: 0.4722
- Basic RAG F1: 0.3324
- Improvement: 42.06%

This script tests H1 on drug recommendation task and compares F1 score.
"""

import torch
import numpy as np
import json
import os
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score

from structured_primitives import StructuredPrimitiveRAG
from models import CausalRAGWithAdjustment
from config import Config
from data import create_dataset_and_loaders

device = torch.device('cpu')
print(f"Using device: {device}")

def compute_f1_metrics(predictions, targets, threshold=0.5):
    """Compute F1, precision, recall for binary classification."""
    # Convert to binary predictions
    pred_binary = (predictions > threshold).astype(int)
    target_binary = (targets > threshold).astype(int)
    
    # Compute metrics
    f1 = f1_score(target_binary, pred_binary, average='binary', zero_division=0)
    precision = precision_score(target_binary, pred_binary, average='binary', zero_division=0)
    recall = recall_score(target_binary, pred_binary, average='binary', zero_division=0)
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_and_evaluate_f1(model, train_loader, test_loader, corpus_embeddings, config, name="Model"):
    """Train model and evaluate F1 score."""
    print(f"\nTraining {name}...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    model.train()
    
    # Training
    for epoch in range(5):
        epoch_loss = 0
        for i, batch in enumerate(train_loader):
            if i >= 10:  # First 10 batches for speed
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()
            pred = model(batch, corpus_embeddings.to(device))
            loss = model.compute_loss(pred, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Epoch {epoch+1}/5, Loss: {epoch_loss/min(10,len(train_loader)):.4f}")
    
    # Evaluation
    print(f"Evaluating {name}...")
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 10:  # First 10 batches for speed
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            pred = model(batch, corpus_embeddings.to(device))
            
            # Get predictions and targets
            predictions = pred['outcome'].cpu().numpy()
            targets = batch['outcome'].cpu().numpy()
            
            all_preds.extend(predictions.flatten())
            all_targets.extend(targets.flatten())
    
    # Compute F1 metrics
    metrics = compute_f1_metrics(np.array(all_preds), np.array(all_targets))
    
    return metrics

def main():
    print("="*60)
    print("H1 vs PACE-RAG COMPARISON (F1 Score)")
    print("="*60)
    
    # Create dataset
    print("Creating dataset...")
    dataset, train_loader, val_loader, test_loader, corpus_embeddings = create_dataset_and_loaders()
    config = dataset.config
    print(f"Dataset created: {len(train_loader)} train batches")
    
    # PACE-RAG benchmark values
    pace_rag_f1 = 0.4722
    basic_rag_f1 = 0.3324
    
    print(f"\nPACE-RAG Benchmark:")
    print(f"  PACE-RAG F1: {pace_rag_f1:.4f}")
    print(f"  Basic RAG F1: {basic_rag_f1:.4f}")
    print(f"  Improvement: {(pace_rag_f1 - basic_rag_f1) / basic_rag_f1 * 100:.2f}%")
    
    # Train and evaluate H1
    print("\n" + "="*50)
    print("H1: Structured Causal Primitives")
    print("="*50)
    
    model_h1 = StructuredPrimitiveRAG(config).to(device)
    h1_metrics = train_and_evaluate_f1(
        model_h1, train_loader, test_loader, corpus_embeddings, config, "H1"
    )
    
    # Train and evaluate baseline
    print("\n" + "="*50)
    print("Baseline: CausalRAGWithAdjustment")
    print("="*50)
    
    baseline = CausalRAGWithAdjustment(config).to(device)
    baseline_metrics = train_and_evaluate_f1(
        baseline, train_loader, test_loader, corpus_embeddings, config, "Baseline"
    )
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    print(f"\n{'Model':<25} {'F1':<8} {'Precision':<10} {'Recall':<8}")
    print("-" * 55)
    print(f"{'PACE-RAG':<25} {pace_rag_f1:<8.4f} {'N/A':<10} {'N/A':<8}")
    print(f"{'Basic RAG':<25} {basic_rag_f1:<8.4f} {'N/A':<10} {'N/A':<8}")
    print(f"{'H1 (Ours)':<25} {h1_metrics['f1']:<8.4f} {h1_metrics['precision']:<10.4f} {h1_metrics['recall']:<8.4f}")
    print(f"{'Baseline':<25} {baseline_metrics['f1']:<8.4f} {baseline_metrics['precision']:<10.4f} {baseline_metrics['recall']:<8.4f}")
    
    # Compute improvements
    h1_vs_baseline = (h1_metrics['f1'] - baseline_metrics['f1']) / baseline_metrics['f1'] * 100 if baseline_metrics['f1'] > 0 else 0
    h1_vs_basic = (h1_metrics['f1'] - basic_rag_f1) / basic_rag_f1 * 100 if basic_rag_f1 > 0 else 0
    h1_vs_pace = (h1_metrics['f1'] - pace_rag_f1) / pace_rag_f1 * 100 if pace_rag_f1 > 0 else 0
    
    print(f"\nH1 vs Baseline: {h1_vs_baseline:+.2f}%")
    print(f"H1 vs Basic RAG: {h1_vs_basic:+.2f}%")
    print(f"H1 vs PACE-RAG: {h1_vs_pace:+.2f}%")
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    if h1_metrics['f1'] > pace_rag_f1:
        print("✓ H1 outperforms PACE-RAG!")
        print(f"  Improvement: {h1_vs_pace:+.2f}%")
    elif h1_metrics['f1'] > basic_rag_f1:
        print("✓ H1 outperforms Basic RAG")
        print(f"  Improvement: {h1_vs_basic:+.2f}%")
        print(f"  But still below PACE-RAG: {h1_vs_pace:+.2f}%")
    else:
        print("✗ H1 underperforms Basic RAG")
        print(f"  Needs improvement")
    
    # Save results
    results = {
        'pace_rag': {
            'f1': pace_rag_f1,
            'source': 'PACE-RAG paper'
        },
        'basic_rag': {
            'f1': basic_rag_f1,
            'source': 'Previous experiments'
        },
        'h1': {
            'f1': h1_metrics['f1'],
            'precision': h1_metrics['precision'],
            'recall': h1_metrics['recall'],
            'vs_pace': h1_vs_pace,
            'vs_basic': h1_vs_basic
        },
        'baseline': {
            'f1': baseline_metrics['f1'],
            'precision': baseline_metrics['precision'],
            'recall': baseline_metrics['recall']
        },
        'timestamp': datetime.now().isoformat()
    }
    
    output_path = os.path.join(os.path.dirname(__file__), 'h1_pace_rag_comparison.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")

if __name__ == '__main__':
    main()
