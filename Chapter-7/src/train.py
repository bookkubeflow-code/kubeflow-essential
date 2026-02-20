#!/usr/bin/env python3
"""
Katib Hyperparameter Tuning Example: Random Forest Classifier

This training script demonstrates proper integration with Katib for
hyperparameter optimization. It implements:

1. Command-line argument parsing for hyperparameters
2. Structured metric logging for Katib's metrics collector
3. Intermediate metrics for early stopping support
4. Cross-validation for robust evaluation

Dataset: Breast Cancer Wisconsin (built into scikit-learn)
Model: Random Forest Classifier
"""

import argparse
import sys
import time
from datetime import datetime

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


def parse_arguments():
    """
    Parse hyperparameters from command-line arguments.
    
    Katib injects hyperparameter values through command-line arguments.
    Each parameter defined in the Experiment YAML maps to an argument here.
    """
    parser = argparse.ArgumentParser(
        description="Train a Random Forest model with hyperparameter tuning via Katib"
    )
    
    # ============================================================
    # Model Selection (Categorical Parameter)
    # ============================================================
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["random_forest", "gradient_boosting"],
        help="Type of ensemble model to use"
    )
    
    # ============================================================
    # Integer Hyperparameters
    # ============================================================
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in the forest (50-300)"
    )
    
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Maximum depth of each tree (3-20). Use 0 for unlimited."
    )
    
    parser.add_argument(
        "--min-samples-split",
        type=int,
        default=2,
        help="Minimum samples required to split an internal node (2-20)"
    )
    
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="Minimum samples required at a leaf node (1-10)"
    )
    
    # ============================================================
    # Float/Double Hyperparameters
    # ============================================================
    parser.add_argument(
        "--max-features-ratio",
        type=float,
        default=0.5,
        help="Ratio of features to consider for best split (0.1-1.0)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for Gradient Boosting (0.01-0.3)"
    )
    
    # ============================================================
    # Categorical Hyperparameters
    # ============================================================
    parser.add_argument(
        "--criterion",
        type=str,
        default="gini",
        choices=["gini", "entropy", "log_loss"],
        help="Function to measure split quality"
    )
    
    parser.add_argument(
        "--bootstrap",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Whether to use bootstrap samples"
    )
    
    # ============================================================
    # Training Configuration (for Hyperband)
    # ============================================================
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs (for iterative evaluation)"
    )
    
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds"
    )
    
    # ============================================================
    # Reproducibility
    # ============================================================
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def log_metric(name: str, value: float):
    """
    Log a metric in Katib-compatible format.
    
    Katib's StdOut metrics collector parses stdout for patterns like:
        metric_name=value
    
    The metric name must match the objectiveMetricName or 
    additionalMetricNames in your Experiment YAML.
    """
    # Ensure consistent formatting for Katib parsing
    print(f"{name}={value:.6f}")
    # Flush to ensure immediate visibility
    sys.stdout.flush()


def log_intermediate_metric(epoch: int, name: str, value: float):
    """
    Log intermediate metrics for early stopping algorithms.
    
    Hyperband and other early stopping algorithms use intermediate
    metrics to decide whether to terminate underperforming trials.
    
    Format: epoch=N metric_name=value
    """
    print(f"epoch={epoch} {name}={value:.6f}")
    sys.stdout.flush()


def log_info(message: str):
    """Log informational messages with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()


def load_and_prepare_data(random_state: int):
    """
    Load the Breast Cancer dataset and prepare train/test splits.
    
    The Breast Cancer Wisconsin dataset is ideal for this example:
    - Binary classification (malignant vs benign)
    - 569 samples, 30 features
    - Well-balanced classes
    - No missing values
    - Built into scikit-learn (no external data needed)
    """
    log_info("Loading Breast Cancer Wisconsin dataset...")
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    log_info(f"Dataset shape: {X.shape}")
    log_info(f"Features: {data.feature_names[:5]}... ({len(data.feature_names)} total)")
    log_info(f"Classes: {data.target_names}")
    log_info(f"Class distribution: {np.bincount(y)}")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=random_state,
        stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    log_info(f"Training samples: {len(X_train)}")
    log_info(f"Test samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def create_model(args):
    """
    Create and configure the model based on hyperparameters.
    
    This function translates command-line arguments into model configuration.
    It handles the conversion of string arguments to proper Python types.
    """
    # Convert max_depth: 0 means unlimited (None)
    max_depth = args.max_depth if args.max_depth > 0 else None
    
    # Convert bootstrap string to boolean
    bootstrap = args.bootstrap.lower() == "true"
    
    # Convert max_features_ratio to actual parameter
    max_features = args.max_features_ratio
    
    log_info("=" * 60)
    log_info("Model Configuration")
    log_info("=" * 60)
    log_info(f"  Model Type: {args.model_type}")
    log_info(f"  n_estimators: {args.n_estimators}")
    log_info(f"  max_depth: {max_depth}")
    log_info(f"  min_samples_split: {args.min_samples_split}")
    log_info(f"  min_samples_leaf: {args.min_samples_leaf}")
    log_info(f"  max_features: {max_features}")
    
    if args.model_type == "random_forest":
        log_info(f"  criterion: {args.criterion}")
        log_info(f"  bootstrap: {bootstrap}")
        
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_features=max_features,
            criterion=args.criterion,
            bootstrap=bootstrap,
            random_state=args.random_state,
            n_jobs=-1,  # Use all available cores
        )
    else:  # gradient_boosting
        log_info(f"  learning_rate: {args.learning_rate}")
        
        model = GradientBoostingClassifier(
            n_estimators=args.n_estimators,
            max_depth=max_depth if max_depth else 3,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_features=max_features,
            learning_rate=args.learning_rate,
            random_state=args.random_state,
        )
    
    log_info("=" * 60)
    
    return model


def train_with_intermediate_metrics(model, X_train, y_train, X_test, y_test, args):
    """
    Train the model and log intermediate metrics.
    
    For tree-based models, we simulate "epochs" using increasing numbers
    of estimators. This enables early stopping algorithms like Hyperband
    to terminate underperforming configurations early.
    
    For real iterative models (neural networks, gradient boosting with 
    staged_predict), you would log actual training progress.
    """
    log_info("Starting training with intermediate metric logging...")
    
    num_epochs = args.num_epochs
    base_estimators = max(1, args.n_estimators // num_epochs)
    
    for epoch in range(1, num_epochs + 1):
        # Calculate number of estimators for this "epoch"
        current_n_estimators = min(epoch * base_estimators, args.n_estimators)
        
        # Update model with current number of estimators
        if hasattr(model, 'n_estimators'):
            model.n_estimators = current_n_estimators
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log intermediate metrics for early stopping
        # Format: epoch=N metric_name=value
        log_intermediate_metric(epoch, "validation-accuracy", accuracy)
        log_intermediate_metric(epoch, "validation-f1", f1)
        
        log_info(f"Epoch {epoch}/{num_epochs}: "
                f"estimators={current_n_estimators}, "
                f"accuracy={accuracy:.4f}, f1={f1:.4f}")
        
        # Small delay to simulate training time
        time.sleep(0.1)
    
    # Ensure final model has full number of estimators
    if hasattr(model, 'n_estimators'):
        model.n_estimators = args.n_estimators
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, args):
    """
    Perform comprehensive model evaluation.
    
    This function:
    1. Runs cross-validation for robust estimates
    2. Evaluates on the held-out test set
    3. Computes multiple metrics
    4. Logs all results in Katib-compatible format
    """
    log_info("=" * 60)
    log_info("Model Evaluation")
    log_info("=" * 60)
    
    # Cross-validation on training data
    log_info(f"Running {args.cv_folds}-fold cross-validation...")
    
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    log_info(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    log_info(f"CV Scores: {cv_scores}")
    
    # Final evaluation on test set
    log_info("Evaluating on test set...")
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate all metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, y_prob)
    
    log_info("-" * 40)
    log_info("Test Set Results:")
    log_info(f"  Accuracy:  {accuracy:.4f}")
    log_info(f"  F1 Score:  {f1:.4f}")
    log_info(f"  Precision: {precision:.4f}")
    log_info(f"  Recall:    {recall:.4f}")
    log_info(f"  AUC-ROC:   {auc:.4f}")
    log_info("-" * 40)
    
    # Feature importance (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_features = np.argsort(importances)[-5:][::-1]
        log_info("Top 5 Important Features (indices):")
        for idx in top_features:
            log_info(f"  Feature {idx}: {importances[idx]:.4f}")
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'cv_accuracy_mean': cv_scores.mean(),
        'cv_accuracy_std': cv_scores.std(),
    }


def main():
    """
    Main training pipeline.
    
    Flow:
    1. Parse hyperparameters from command line
    2. Load and prepare data
    3. Create model with specified hyperparameters
    4. Train with intermediate metrics (for early stopping)
    5. Evaluate comprehensively
    6. Log final metrics for Katib
    """
    log_info("=" * 60)
    log_info("Katib Hyperparameter Tuning - Training Script")
    log_info("=" * 60)
    
    # Parse command-line arguments (hyperparameters from Katib)
    args = parse_arguments()
    
    # Set random seed for reproducibility
    np.random.seed(args.random_state)
    
    try:
        # Load and prepare data
        X_train, X_test, y_train, y_test = load_and_prepare_data(args.random_state)
        
        # Create model with hyperparameters
        model = create_model(args)
        
        # Train with intermediate metrics for early stopping support
        model = train_with_intermediate_metrics(
            model, X_train, y_train, X_test, y_test, args
        )
        
        # Comprehensive evaluation
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test, args)
        
        # ============================================================
        # LOG FINAL METRICS FOR KATIB
        # ============================================================
        # These are the metrics Katib uses for optimization.
        # The format MUST be: metric_name=value
        # 
        # CRITICAL: These must match your Experiment YAML:
        #   - objectiveMetricName
        #   - additionalMetricNames
        # ============================================================
        
        log_info("=" * 60)
        log_info("FINAL METRICS (Katib Collection)")
        log_info("=" * 60)
        
        # Primary objective metric
        log_metric("accuracy", metrics['accuracy'])
        
        # Additional metrics for tracking
        log_metric("f1_score", metrics['f1_score'])
        log_metric("precision", metrics['precision'])
        log_metric("recall", metrics['recall'])
        log_metric("auc", metrics['auc'])
        log_metric("cv_accuracy", metrics['cv_accuracy_mean'])
        
        log_info("=" * 60)
        log_info("Training completed successfully!")
        log_info("=" * 60)
        
    except Exception as e:
        log_info(f"ERROR: Training failed with exception: {e}")
        # Log a failure metric so Katib knows the trial failed
        log_metric("accuracy", 0.0)
        sys.exit(1)


if __name__ == "__main__":
    main()

