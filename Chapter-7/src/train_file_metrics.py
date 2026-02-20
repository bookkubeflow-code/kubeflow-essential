#!/usr/bin/env python3
"""
Katib Hyperparameter Tuning Example: File-Based Metrics Collection

This is an alternative training script that writes metrics to a file
instead of stdout. Use this with Katib's File metrics collector.

Usage with File Collector:
    metricsCollectorSpec:
      collector:
        kind: File
      source:
        fileSystemPath:
          path: /var/log/katib/metrics.log
          kind: File
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


# Metrics file path - must match the path in your Experiment YAML
METRICS_FILE = os.environ.get('KATIB_METRICS_FILE', '/var/log/katib/metrics.log')


def parse_arguments():
    """Parse hyperparameters from command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Random Forest model with file-based metrics"
    )
    
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--min-samples-split', type=int, default=2)
    parser.add_argument('--min-samples-leaf', type=int, default=1)
    parser.add_argument('--max-features-ratio', type=float, default=0.5)
    parser.add_argument('--criterion', type=str, default='gini',
                       choices=['gini', 'entropy', 'log_loss'])
    parser.add_argument('--bootstrap', type=str, default='true',
                       choices=['true', 'false'])
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--cv-folds', type=int, default=5)
    parser.add_argument('--random-state', type=int, default=42)
    
    return parser.parse_args()


class FileMetricsLogger:
    """
    Logger that writes metrics to a file for Katib's File collector.
    
    This class handles:
    - Creating the metrics file directory if needed
    - Writing metrics in the correct format
    - Flushing after each write to ensure immediate visibility
    """
    
    def __init__(self, filepath: str = METRICS_FILE):
        self.filepath = filepath
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Create the metrics directory if it doesn't exist."""
        directory = os.path.dirname(self.filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    
    def log_metric(self, name: str, value: float):
        """
        Log a metric to the file.
        
        Format: metric_name=value
        This must match your Experiment's metricsCollectorSpec filter.
        """
        with open(self.filepath, 'a') as f:
            f.write(f"{name}={value:.6f}\n")
            f.flush()
        # Also print to stdout for debugging
        print(f"[Metric logged] {name}={value:.6f}")
    
    def log_intermediate_metric(self, epoch: int, name: str, value: float):
        """
        Log intermediate metric with epoch information.
        
        Format: epoch=N metric_name=value
        Used for early stopping algorithms.
        """
        with open(self.filepath, 'a') as f:
            f.write(f"epoch={epoch} {name}={value:.6f}\n")
            f.flush()
        print(f"[Intermediate] epoch={epoch} {name}={value:.6f}")


def log_info(message: str):
    """Log informational messages with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()


def main():
    """Main training pipeline with file-based metrics logging."""
    log_info("=" * 60)
    log_info("Katib Training - File-Based Metrics Collection")
    log_info("=" * 60)
    
    args = parse_arguments()
    np.random.seed(args.random_state)
    
    # Initialize file-based metrics logger
    logger = FileMetricsLogger()
    log_info(f"Metrics will be written to: {logger.filepath}")
    
    try:
        # Load data
        log_info("Loading dataset...")
        data = load_breast_cancer()
        X, y = data.data, data.target
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.random_state, stratify=y
        )
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Configure model
        max_depth = args.max_depth if args.max_depth > 0 else None
        bootstrap = args.bootstrap.lower() == 'true'
        
        log_info(f"Configuration: n_estimators={args.n_estimators}, "
                f"max_depth={max_depth}, criterion={args.criterion}")
        
        # Train with intermediate metrics
        num_epochs = args.num_epochs
        base_estimators = max(1, args.n_estimators // num_epochs)
        
        model = RandomForestClassifier(
            max_depth=max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features_ratio,
            criterion=args.criterion,
            bootstrap=bootstrap,
            random_state=args.random_state,
            n_jobs=-1,
        )
        
        for epoch in range(1, num_epochs + 1):
            current_n_estimators = min(epoch * base_estimators, args.n_estimators)
            model.n_estimators = current_n_estimators
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Log intermediate metrics to file
            logger.log_intermediate_metric(epoch, "validation-accuracy", accuracy)
            logger.log_intermediate_metric(epoch, "validation-f1", f1)
        
        # Final evaluation
        model.n_estimators = args.n_estimators
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        final_accuracy = accuracy_score(y_test, y_pred)
        final_f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, 
                            random_state=args.random_state)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        
        # Log final metrics to file
        log_info("=" * 40)
        log_info("FINAL METRICS (File-Based)")
        log_info("=" * 40)
        
        logger.log_metric("accuracy", final_accuracy)
        logger.log_metric("f1_score", final_f1)
        logger.log_metric("cv_accuracy", cv_scores.mean())
        
        log_info(f"Final Accuracy: {final_accuracy:.4f}")
        log_info(f"Final F1 Score: {final_f1:.4f}")
        log_info(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        log_info("Training completed successfully!")
        
    except Exception as e:
        log_info(f"ERROR: Training failed: {e}")
        logger.log_metric("accuracy", 0.0)
        sys.exit(1)


if __name__ == "__main__":
    main()

