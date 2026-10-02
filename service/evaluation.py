import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from typing import List, Tuple

def evaluate_model(predictions: np.ndarray, ground_truth: np.ndarray) -> dict:
    """Score model predictions against labeled data."""
    pred_labels = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(ground_truth, pred_labels)
    f1 = f1_score(ground_truth, pred_labels, average='weighted', zero_division=0)
    
    return {
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'num_samples': len(ground_truth),
    }

def run_evaluation(preds: List[List[float]], labels: List[int]) -> dict:
    """Wrapper for easy eval execution."""
    pred_array = np.array(preds)
    label_array = np.array(labels)
    return evaluate_model(pred_array, label_array)
