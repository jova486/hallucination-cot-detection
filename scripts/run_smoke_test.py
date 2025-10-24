#!/usr/bin/env python3
"""Run smoke test from CLI."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from generator import IncrementalCoTGenerator
from verifier import self_annotate_dataset_robust

def compute_simple_features(X):
    n, L, d = X.shape
    features = []
    norms = np.linalg.norm(X, axis=2)
    features.append(norms.mean(axis=1))
    features.append(norms.std(axis=1))
    u = X / (np.linalg.norm(X, axis=2, keepdims=True) + 1e-12)
    theta = np.linalg.norm(u[:, 1:, :] - u[:, :-1, :], axis=2)
    features.append(theta.mean(axis=1))
    features.append(theta.std(axis=1))
    features.append(theta.max(axis=1))
    return np.column_stack(features)

def main():
    print("="*80)
    print("🔥 SMOKE TEST")
    print("="*80)
    
    OUT = Path("./outputs/smoke_test")
    OUT.mkdir(parents=True, exist_ok=True)
    
    print("\n[1/5] Loading dataset...")
    ds = load_dataset("gsm8k", "main", split="train[:100]")
    
    print("\n[2/5] Generating CoT...")
    gen = IncrementalCoTGenerator("meta-llama/Meta-Llama-3-8B-Instruct")
    
    for i, ex in enumerate(tqdm(ds)):
        result = gen.generate_cot_with_step_activations(ex["question"], ex["answer"])
        np.savez_compressed(
            OUT / f"ex_{i:03d}.npz",
            question=result["question"],
            gold_answer=result["gold_answer"],
            steps_text=[s["step_text"] for s in result["steps"]],
            activations_last=np.stack([s["activations_last"] for s in result["steps"]]),
            mean_logprobs=np.array([s["mean_logprob"] for s in result["steps"]])
        )
    
    print("\n[3/5] Annotating...")
    anns, _, _ = self_annotate_dataset_robust(str(OUT), str(OUT / "annotations.json"))
    
    print("\n[4/5] Loading data...")
    X_list, y_list, lp_list = [], [], []
    for a in anns:
        if a["label"] == -1:
            continue
        d = np.load(a["file"], allow_pickle=True)
        X_list.append(d["activations_last"][a["step_idx"]])
        y_list.append(a["label"])
        lp_list.append(d["mean_logprobs"][a["step_idx"]])
    
    X = np.stack(X_list)
    y = np.array(y_list)
    lp = np.array(lp_list)
    print(f"Shape: {X.shape}, Positive: {y.sum()}/{len(y)}")
    
    print("\n[5/5] Training...")
    baseline = LogisticRegression(max_iter=1000, random_state=42)
    base_auc = cross_val_score(baseline, lp.reshape(-1,1), y, cv=5, scoring="roc_auc").mean()
    
    feats = compute_simple_features(X)
    model = make_pipeline(StandardScaler(), 
                         LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced"))
    your_auc = cross_val_score(model, feats, y, cv=5, scoring="roc_auc").mean()
    
    delta = your_auc - base_auc
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Baseline: {base_auc:.4f}")
    print(f"Your model: {your_auc:.4f}")
    print(f"Δ: {delta:+.4f}")
    print()
    print("✅ PASS" if delta > 0.03 else "⚠️  MARGINAL" if delta > 0 else "❌ FAIL")
    print("="*80)

if __name__ == "__main__":
    main()
