"""
Evaluation Script for Siamese Network Signature Verification

This script evaluates trained models on:
- Accuracy metrics
- ROC curve and AUC
- Precision, Recall, F1-Score
- Distance distributions
- Visualization of results
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                              f1_score, roc_curve, auc, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import config
from model import create_siamese_model
from dataset import get_data_loaders
from utils import load_checkpoint


class ModelEvaluator:
    """
    Evaluator class for Siamese Networks.
    """
    
    def __init__(self, model, device, test_loader, config):
        """
        Args:
            model: Trained Siamese network
            device: Device to use
            test_loader: Test data loader
            config: Configuration module
        """
        self.model = model
        self.device = device
        self.test_loader = test_loader
        self.config = config
        
        self.model.eval()
    
    def compute_distances_and_labels(self):
        """
        Compute distances and labels for all test pairs.
        
        Returns:
            tuple: (distances, labels) as numpy arrays
        """
        distances = []
        labels = []
        
        print("Computing distances on test set...")
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                if len(batch) == 3:  # Pair dataset
                    img1, img2, label = batch
                    img1 = img1.to(self.device)
                    img2 = img2.to(self.device)
                    
                    # Get embeddings
                    if hasattr(self.model, 'forward'):
                        emb1, emb2 = self.model(img1, img2)
                    else:
                        emb1 = self.model.embedding_network(img1)
                        emb2 = self.model.embedding_network(img2)
                    
                    # Compute Euclidean distances
                    dist = F.pairwise_distance(emb1, emb2, p=2)
                    
                    distances.extend(dist.cpu().numpy())
                    labels.extend(label.cpu().numpy())
        
        return np.array(distances), np.array(labels)
    
    def find_optimal_threshold(self, distances, labels):
        """
        Find optimal threshold that maximizes accuracy.
        
        Args:
            distances (np.ndarray): Distance values
            labels (np.ndarray): True labels (0=genuine, 1=impostor)
        
        Returns:
            tuple: (optimal_threshold, best_accuracy)
        """
        thresholds = np.linspace(distances.min(), distances.max(), 100)
        accuracies = []
        
        for threshold in thresholds:
            predictions = (distances > threshold).astype(int)
            acc = accuracy_score(labels, predictions)
            accuracies.append(acc)
        
        best_idx = np.argmax(accuracies)
        optimal_threshold = thresholds[best_idx]
        best_accuracy = accuracies[best_idx]
        
        return optimal_threshold, best_accuracy
    
    def evaluate_with_threshold(self, distances, labels, threshold):
        """
        Evaluate model performance with a given threshold.
        
        Args:
            distances (np.ndarray): Distance values
            labels (np.ndarray): True labels
            threshold (float): Distance threshold
        
        Returns:
            dict: Dictionary of metrics
        """
        predictions = (distances > threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1_score': f1_score(labels, predictions, zero_division=0),
            'confusion_matrix': confusion_matrix(labels, predictions)
        }
        
        return metrics
    
    def compute_roc_curve(self, distances, labels):
        """
        Compute ROC curve and AUC.
        
        Args:
            distances (np.ndarray): Distance values
            labels (np.ndarray): True labels
        
        Returns:
            tuple: (fpr, tpr, thresholds, auc_score)
        """
        # For ROC curve, we need scores where higher = more likely impostor
        # So we use distances directly (higher distance = more likely impostor)
        fpr, tpr, thresholds = roc_curve(labels, distances)
        auc_score = auc(fpr, tpr)
        
        return fpr, tpr, thresholds, auc_score
    
    def evaluate(self):
        """
        Complete evaluation of the model.
        
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        # Compute distances and labels
        distances, labels = self.compute_distances_and_labels()
        
        # Find optimal threshold
        optimal_threshold, best_accuracy = self.find_optimal_threshold(distances, labels)
        
        print(f"\nOptimal threshold: {optimal_threshold:.4f}")
        print(f"Best accuracy: {best_accuracy:.4f}")
        
        # Evaluate with optimal threshold
        metrics = self.evaluate_with_threshold(distances, labels, optimal_threshold)
        
        # Compute ROC curve
        fpr, tpr, roc_thresholds, auc_score = self.compute_roc_curve(distances, labels)
        
        # Compile results
        results = {
            'distances': distances,
            'labels': labels,
            'optimal_threshold': optimal_threshold,
            'best_accuracy': best_accuracy,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'confusion_matrix': metrics['confusion_matrix'],
            'fpr': fpr,
            'tpr': tpr,
            'roc_thresholds': roc_thresholds,
            'auc': auc_score
        }
        
        return results
    
    def print_results(self, results):
        """
        Print evaluation results.
        
        Args:
            results (dict): Results dictionary from evaluate()
        """
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        print(f"\nOptimal Threshold: {results['optimal_threshold']:.4f}")
        print(f"\nAccuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1-Score:  {results['f1_score']:.4f}")
        print(f"AUC:       {results['auc']:.4f}")
        
        print("\nConfusion Matrix:")
        print("                 Predicted")
        print("               Genuine  Impostor")
        print(f"Actual Genuine    {results['confusion_matrix'][0, 0]:6d}    {results['confusion_matrix'][0, 1]:6d}")
        print(f"       Impostor   {results['confusion_matrix'][1, 0]:6d}    {results['confusion_matrix'][1, 1]:6d}")
        
        # Calculate error rates
        tn, fp, fn, tp = results['confusion_matrix'].ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Accept Rate
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Reject Rate
        
        print(f"\nFalse Accept Rate (FAR):  {far:.4f}")
        print(f"False Reject Rate (FRR):  {frr:.4f}")
        
        print("="*80 + "\n")
    
    def plot_results(self, results, save_dir=None):
        """
        Plot evaluation results.
        
        Args:
            results (dict): Results dictionary from evaluate()
            save_dir (str): Directory to save plots (None to just show)
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Distance distribution
        ax1 = plt.subplot(2, 3, 1)
        genuine_mask = results['labels'] == 0
        impostor_mask = results['labels'] == 1
        
        plt.hist(results['distances'][genuine_mask], bins=50, alpha=0.6, 
                 label='Genuine Pairs', density=True)
        plt.hist(results['distances'][impostor_mask], bins=50, alpha=0.6, 
                 label='Impostor Pairs', density=True)
        plt.axvline(results['optimal_threshold'], color='r', linestyle='--', 
                    label=f'Threshold={results["optimal_threshold"]:.3f}')
        plt.xlabel('Distance')
        plt.ylabel('Density')
        plt.title('Distance Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. ROC Curve
        ax2 = plt.subplot(2, 3, 2)
        plt.plot(results['fpr'], results['tpr'], linewidth=2, 
                 label=f'ROC Curve (AUC = {results["auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Confusion Matrix
        ax3 = plt.subplot(2, 3, 3)
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Genuine', 'Impostor'],
                    yticklabels=['Genuine', 'Impostor'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        # 4. Precision-Recall vs Threshold
        ax4 = plt.subplot(2, 3, 4)
        thresholds = np.linspace(results['distances'].min(), 
                                 results['distances'].max(), 100)
        precisions = []
        recalls = []
        
        for threshold in thresholds:
            preds = (results['distances'] > threshold).astype(int)
            precisions.append(precision_score(results['labels'], preds, zero_division=0))
            recalls.append(recall_score(results['labels'], preds, zero_division=0))
        
        plt.plot(thresholds, precisions, label='Precision', linewidth=2)
        plt.plot(thresholds, recalls, label='Recall', linewidth=2)
        plt.axvline(results['optimal_threshold'], color='r', linestyle='--', 
                    label='Optimal Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Precision and Recall vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Accuracy vs Threshold
        ax5 = plt.subplot(2, 3, 5)
        accuracies = []
        for threshold in thresholds:
            preds = (results['distances'] > threshold).astype(int)
            accuracies.append(accuracy_score(results['labels'], preds))
        
        plt.plot(thresholds, accuracies, linewidth=2)
        plt.axvline(results['optimal_threshold'], color='r', linestyle='--', 
                    label=f'Optimal (Acc={results["accuracy"]:.3f})')
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Error Rates vs Threshold
        ax6 = plt.subplot(2, 3, 6)
        fars = []
        frrs = []
        
        for threshold in thresholds:
            preds = (results['distances'] > threshold).astype(int)
            cm = confusion_matrix(results['labels'], preds)
            tn, fp, fn, tp = cm.ravel()
            
            far = fp / (fp + tn) if (fp + tn) > 0 else 0
            frr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            fars.append(far)
            frrs.append(frr)
        
        plt.plot(thresholds, fars, label='FAR (False Accept Rate)', linewidth=2)
        plt.plot(thresholds, frrs, label='FRR (False Reject Rate)', linewidth=2)
        plt.axvline(results['optimal_threshold'], color='r', linestyle='--', 
                    label='Optimal Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Error Rate')
        plt.title('Error Rates vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'evaluation_results.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Results plot saved to {save_dir}")
        
        plt.show()


def main():
    """Main evaluation function."""
    print("="*80)
    print("Model Evaluation")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    model = create_siamese_model(config, loss_type=config.LOSS_TYPE)
    model = model.to(config.DEVICE)
    
    # Load best checkpoint
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: No checkpoint found at {checkpoint_path}")
        print("Please train the model first!")
        return
    
    checkpoint = load_checkpoint(checkpoint_path, model)
    
    # Create test data loader
    print("\nLoading test data...")
    _, test_loader = get_data_loaders(config, loss_type='contrastive')  # Use pair loader for eval
    
    # Create evaluator
    evaluator = ModelEvaluator(model, config.DEVICE, test_loader, config)
    
    # Run evaluation
    print("\nEvaluating model...")
    results = evaluator.evaluate()
    
    # Print results
    evaluator.print_results(results)
    
    # Plot results
    print("\nGenerating plots...")
    evaluator.plot_results(results, save_dir=config.RESULTS_DIR)
    
    # Save results
    results_file = os.path.join(config.RESULTS_DIR, 'evaluation_metrics.npz')
    np.savez(results_file, **results)
    print(f"\nResults saved to {results_file}")
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
