from kfp import dsl
from kfp.dsl import Dataset, Model
from typing import NamedTuple

@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def load_data(dataset_url: str, output_dataset: dsl.Output[Dataset]):
    import pandas as pd
    import os
    
    # Load the dataset
    df = pd.read_csv(dataset_url)
    
    # Basic data cleaning
    df = df.dropna()
    
    # Save the dataset to the output artifact path
    df.to_csv(output_dataset.path, index=False)
    
    print(f"Dataset loaded and saved to {output_dataset.path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Set metadata
    output_dataset.metadata = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns)
    }

@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def train_model(
    input_dataset: dsl.Input[Dataset], 
    model_output: dsl.Output[Model]
) -> NamedTuple('Outputs', [('accuracy', float)]):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import joblib
    import os
    
    # Load data from input artifact
    df = pd.read_csv(input_dataset.path)
    
    # Prepare features and target
    feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = df[feature_columns]
    y = df['species']
    
    print(f"Training with {len(df)} samples")
    print(f"Features: {feature_columns}")
    print(f"Target classes: {y.unique()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Training completed. Accuracy: {accuracy:.4f}")
    
    # Save model to output artifact
    joblib.dump(model, model_output.path)
    
    # Set model metadata
    model_output.metadata = {
        "accuracy": accuracy,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": feature_columns,
        "model_type": "LogisticRegression"
    }
    
    return (accuracy,)

@dsl.component(
    base_image="python:3.9", 
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def evaluate_model(
    test_dataset: dsl.Input[Dataset], 
    model_input: dsl.Input[Model]
) -> NamedTuple('Outputs', [('final_accuracy', float), ('classification_report', str)]):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import joblib
    
    # Load data from input artifact
    df = pd.read_csv(test_dataset.path)
    
    # Prepare features and target
    feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = df[feature_columns]
    y = df['species']
    
    print(f"Evaluating with {len(df)} samples")
    print(f"Features: {feature_columns}")
    
    # Split data (same split as training for consistency)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load saved model from input artifact
    model = joblib.load(model_input.path)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Final Model Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{report}")
    
    # Also evaluate on full dataset for completeness
    y_full_pred = model.predict(X)
    full_accuracy = accuracy_score(y, y_full_pred)
    print(f"Full Dataset Accuracy: {full_accuracy:.4f}")
    
    return (accuracy, report) 