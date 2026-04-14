"""
Simplified hypothesis test - just test H1 with baseline comparison.
"""

import torch
import numpy as np
import json
import os
from datetime import datetime

# Import implementations
from structured_primitives import StructuredPrimitiveRAG
from models import CausalRAGWithAdjustment
from config import Config
from data import create_dataset_and_loaders

device = torch.device('cpu')
print(f"Using device: {device}")

# Create dataset once
print("Creating dataset...")
dataset, train_loader, val_loader, test_loader, corpus_embeddings = create_dataset_and_loaders()
print(f"Dataset created: {len(train_loader)} train batches, {len(test_loader)} test batches")
print(f"Corpus embeddings shape: {corpus_embeddings.shape}")

# Use the actual config from dataset (embedding_dim may be overridden by sentence-transformers)
config = dataset.config
print(f"Actual embedding_dim: {config.embedding_dim}")

results = {}

# Test H1: Structured Primitives
print("\n" + "="*50)
print("H1: Structured Causal Primitives")
print("="*50)

model_h1 = StructuredPrimitiveRAG(config).to(device)
optimizer = torch.optim.Adam(model_h1.parameters(), lr=config.lr)

print("Training H1 model...")
model_h1.train()
for epoch in range(5):
    epoch_loss = 0
    for i, batch in enumerate(train_loader):
        if i >= 10:  # Only first 10 batches for speed
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        optimizer.zero_grad()
        pred = model_h1(batch, corpus_embeddings.to(device))
        loss = model_h1.compute_loss(pred, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"  Epoch {epoch+1}/5, Loss: {epoch_loss/min(10,len(train_loader)):.4f}")

# Evaluate H1
print("\nEvaluating H1...")
model_h1.eval()
h1_loss = 0
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        if i >= 10:  # Only first 10 batches for speed
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        pred = model_h1(batch, corpus_embeddings.to(device))
        loss = model_h1.compute_loss(pred, batch)
        h1_loss += loss.item()

h1_loss = h1_loss / min(10, len(test_loader))
print(f"  H1 Test Loss: {h1_loss:.4f}")

# Train baseline
print("\nTraining baseline model...")
baseline = CausalRAGWithAdjustment(config).to(device)
baseline_optimizer = torch.optim.Adam(baseline.parameters(), lr=config.lr)

baseline.train()
for epoch in range(5):
    epoch_loss = 0
    for i, batch in enumerate(train_loader):
        if i >= 10:  # Only first 10 batches for speed
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        baseline_optimizer.zero_grad()
        pred = baseline(batch, corpus_embeddings.to(device))
        loss = baseline.compute_loss(pred, batch)
        loss.backward()
        baseline_optimizer.step()
        epoch_loss += loss.item()
    print(f"  Epoch {epoch+1}/5, Loss: {epoch_loss/min(10,len(train_loader)):.4f}")

# Evaluate baseline
print("\nEvaluating baseline...")
baseline.eval()
baseline_loss = 0
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        if i >= 10:  # Only first 10 batches for speed
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        pred = baseline(batch, corpus_embeddings.to(device))
        loss = baseline.compute_loss(pred, batch)
        baseline_loss += loss.item()

baseline_loss = baseline_loss / min(10, len(test_loader))
print(f"  Baseline Test Loss: {baseline_loss:.4f}")

# Compare
improvement = (baseline_loss - h1_loss) / baseline_loss * 100 if baseline_loss > 0 else 0

results['H1'] = {
    'structured_primitive_loss': h1_loss,
    'baseline_loss': baseline_loss,
    'improvement_percent': improvement,
    'supported': improvement > 0
}

print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"H1 Loss: {h1_loss:.4f}")
print(f"Baseline Loss: {baseline_loss:.4f}")
print(f"Improvement: {improvement:.2f}%")
print(f"Hypothesis Supported: {results['H1']['supported']}")

# Save results
output_path = os.path.join(os.path.dirname(__file__), 'hypothesis_validation_results.json')
results['timestamp'] = datetime.now().isoformat()
results['summary'] = {
    'H1_supported': results['H1']['supported'],
    'total_supported': 1 if results['H1']['supported'] else 0
}

with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {output_path}")
