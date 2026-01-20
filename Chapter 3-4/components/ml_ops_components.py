from kfp import dsl
from kfp.dsl import Dataset, Model, HTML, Metrics
from typing import NamedTuple

@dsl.component(
    base_image="python:3.11.13",
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def load_data(dataset_url: str, output_dataset: dsl.Output[Dataset]):
    """Load and prepare data from URL."""
    import pandas as pd
    import os
    
    # Load data
    df = pd.read_csv(dataset_url)
    
    # Basic cleaning
    df = df.dropna()
    
    print(f"Loaded {len(df)} samples with {len(df.columns)} features")
    print(f"Columns: {list(df.columns)}")
    
    # Save to output artifact
    df.to_csv(output_dataset.path, index=False)
    
    # Add metadata
    output_dataset.metadata = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns)
    }

@dsl.component(
    base_image="python:3.11.13",
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def train_model(
    input_dataset: dsl.Input[Dataset], 
    model_output: dsl.Output[Model]
) -> NamedTuple('ModelOutputs', [('training_accuracy', float), ('feature_count', int)]):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import joblib
    
    # Load data from input artifact
    df = pd.read_csv(input_dataset.path)
    
    # Prepare features and target (assuming Iris dataset structure)
    feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = df[feature_columns]
    y = df['species']
    
    print(f"Training with {len(df)} samples")
    print(f"Features: {feature_columns}")
    print(f"Classes: {y.unique()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate training accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Training completed. Accuracy: {accuracy:.4f}")
    
    # Save model to output artifact
    joblib.dump(model, model_output.path)
    
    # Add model metadata
    model_output.metadata = {
        "accuracy": accuracy,
        "n_estimators": 100,
        "feature_count": len(feature_columns),
        "training_samples": len(X_train),
        "test_samples": len(X_test)
    }
    
    return NamedTuple('ModelOutputs', [('training_accuracy', float), ('feature_count', int)])(accuracy, len(feature_columns))

@dsl.component(
    base_image="python:3.11.13",
    packages_to_install=[
        "pandas", 
        "scikit-learn", 
        "joblib", 
        "matplotlib==3.7.2", 
        "seaborn==0.12.2", 
        "plotly==5.17.0"
    ]
)
def evaluate_model(
    test_dataset: dsl.Input[Dataset], 
    model_input: dsl.Input[Model],
    model_metrics: dsl.Output[Metrics],
    performance_dashboard: dsl.Output[HTML]
) -> NamedTuple('EvaluationOutputs', [('final_accuracy', float), ('precision', float), ('recall', float), ('f1_score', float)]):
    """Enhanced model evaluation with rich visualizations and interactive dashboard."""
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        precision_recall_fscore_support, roc_curve, auc
    )
    from sklearn.preprocessing import LabelEncoder, label_binarize
    import joblib
    import json
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import matplotlib.pyplot as plt
    import seaborn as sns
    from io import StringIO
    import base64
    from datetime import datetime
    
    print("🔬 Starting Enhanced Model Evaluation with Visualizations")
    
    # Load data from input artifact
    df = pd.read_csv(test_dataset.path)
    
    # Prepare features and target
    feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = df[feature_columns]
    y = df['species']
    
    print(f"📊 Evaluating with {len(df)} samples")
    print(f"🔍 Features: {feature_columns}")
    print(f"🎯 Classes: {y.unique()}")
    
    # Split data (same split as training for consistency)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load saved model from input artifact
    model = joblib.load(model_input.path)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    class_names = model.classes_
    
    # Feature importance
    feature_importance = dict(zip(feature_columns, model.feature_importances_))
    
    print(f"✅ Model Performance Metrics:")
    print(f"   📈 Accuracy: {accuracy:.4f}")
    print(f"   📈 Precision: {precision:.4f}")
    print(f"   📈 Recall: {recall:.4f}")
    print(f"   📈 F1-Score: {f1:.4f}")
    
    # Prepare metrics for Kubeflow UI
    kubeflow_metrics = {
        'metrics': [
            {'name': 'accuracy', 'numberValue': float(accuracy), 'format': 'PERCENTAGE'},
            {'name': 'precision', 'numberValue': float(precision), 'format': 'PERCENTAGE'},
            {'name': 'recall', 'numberValue': float(recall), 'format': 'PERCENTAGE'},
            {'name': 'f1_score', 'numberValue': float(f1), 'format': 'PERCENTAGE'}
        ]
    }
    
    # Save metrics
    with open(model_metrics.path, 'w') as f:
        json.dump(kubeflow_metrics, f, indent=2)
    
    # Create rich HTML dashboard
    print("🎨 Creating Interactive Performance Dashboard...")
    
    # Start HTML document
    html_content = StringIO()
    html_content.write(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Performance Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
            }}
            .dashboard-container {{
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                max-width: 1200px;
                margin: 0 auto;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 3px solid #667eea;
            }}
            .header h1 {{
                color: #667eea;
                font-size: 2.5em;
                margin: 0;
            }}
            .timestamp {{
                color: #666;
                font-size: 0.9em;
                margin-top: 5px;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .metric-card {{
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                padding: 25px;
                border-radius: 12px;
                text-align: center;
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
                transition: transform 0.3s ease;
            }}
            .metric-card:hover {{
                transform: translateY(-5px);
            }}
            .metric-card h3 {{
                margin: 0 0 10px 0;
                font-size: 1.1em;
                opacity: 0.9;
            }}
            .metric-value {{
                font-size: 2.2em;
                font-weight: bold;
                margin: 0;
            }}
            .chart-container {{
                margin: 30px 0;
                background: #f8f9fa;
                padding: 25px;
                border-radius: 12px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            }}
            .chart-title {{
                font-size: 1.4em;
                font-weight: bold;
                color: #667eea;
                margin-bottom: 15px;
                text-align: center;
            }}
            .feature-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }}
            .feature-item {{
                background: white;
                padding: 15px;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }}
        </style>
    </head>
    <body>
        <div class="dashboard-container">
            <div class="header">
                <h1>🤖 ML Model Performance Dashboard</h1>
                <div class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            </div>
    """)
    
    # Add metrics cards
    html_content.write('<div class="metrics-grid">')
    metrics_data = [
        ('Accuracy', accuracy, '📈'),
        ('Precision', precision, '🎯'),
        ('Recall', recall, '🔍'),
        ('F1-Score', f1, '⚖️')
    ]
    
    for name, value, emoji in metrics_data:
        html_content.write(f"""
        <div class="metric-card">
            <h3>{emoji} {name}</h3>
            <div class="metric-value">{value:.1%}</div>
        </div>
        """)
    html_content.write('</div>')
    
    # Create confusion matrix heatmap
    html_content.write('<div class="chart-container">')
    html_content.write('<div class="chart-title">📊 Confusion Matrix</div>')
    
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale='Blues',
        showscale=True
    ))
    fig_cm.update_layout(
        title=None,
        xaxis_title='Predicted Class',
        yaxis_title='Actual Class',
        font=dict(size=12),
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    html_content.write(fig_cm.to_html(div_id="confusion-matrix", include_plotlyjs=False))
    html_content.write('</div>')
    
    # Feature importance chart
    html_content.write('<div class="chart-container">')
    html_content.write('<div class="chart-title">🌟 Feature Importance</div>')
    
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    feature_names, importances = zip(*sorted_features)
    
    fig_importance = go.Figure(go.Bar(
        x=list(importances),
        y=list(feature_names),
        orientation='h',
        marker_color='rgba(102, 126, 234, 0.8)',
        text=[f'{imp:.3f}' for imp in importances],
        textposition='outside'
    ))
    fig_importance.update_layout(
        title=None,
        xaxis_title='Importance Score',
        yaxis_title='Features',
        height=300,
        margin=dict(l=150, r=50, t=50, b=50)
    )
    
    html_content.write(fig_importance.to_html(div_id="feature-importance", include_plotlyjs=False))
    html_content.write('</div>')
    
    # Classification report details
    html_content.write('<div class="chart-container">')
    html_content.write('<div class="chart-title">📋 Detailed Classification Report</div>')
    
    # Create per-class metrics visualization
    classes = [cls for cls in report.keys() if cls not in ['accuracy', 'macro avg', 'weighted avg']]
    metrics_names = ['precision', 'recall', 'f1-score']
    
    fig_report = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Precision', 'Recall', 'F1-Score'),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    colors = ['#667eea', '#764ba2', '#f093fb']
    
    for i, metric in enumerate(metrics_names):
        values = [report[cls][metric] for cls in classes]
        fig_report.add_trace(
            go.Bar(x=classes, y=values, name=metric.title(), 
                  marker_color=colors[i], text=[f'{v:.3f}' for v in values],
                  textposition='outside'),
            row=1, col=i+1
        )
    
    fig_report.update_layout(
        height=400,
        showlegend=False,
        title=None,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    html_content.write(fig_report.to_html(div_id="classification-report", include_plotlyjs=False))
    html_content.write('</div>')
    
    # Model info section
    html_content.write(f"""
        <div class="chart-container">
            <div class="chart-title">🔧 Model Information</div>
            <div class="feature-grid">
                <div class="feature-item">
                    <strong>Model Type:</strong> Random Forest Classifier
                </div>
                <div class="feature-item">
                    <strong>Training Samples:</strong> {len(X_train)}
                </div>
                <div class="feature-item">
                    <strong>Test Samples:</strong> {len(X_test)}
                </div>
                <div class="feature-item">
                    <strong>Features:</strong> {len(feature_columns)}
                </div>
                <div class="feature-item">
                    <strong>Classes:</strong> {len(class_names)}
                </div>
                <div class="feature-item">
                    <strong>Best Feature:</strong> {max(feature_importance, key=feature_importance.get)}
                </div>
            </div>
        </div>
    """)
    
    # Close HTML
    html_content.write("""
        </div>
    </body>
    </html>
    """)
    
    # Save HTML dashboard
    with open(performance_dashboard.path, 'w') as f:
        f.write(html_content.getvalue())
    
    print("✅ Interactive dashboard created successfully!")
    print(f"📊 Dashboard saved to: {performance_dashboard.path}")
    
    # Also save full evaluation on complete dataset
    y_full_pred = model.predict(X)
    full_accuracy = accuracy_score(y, y_full_pred)
    print(f"📈 Full Dataset Accuracy: {full_accuracy:.4f}")
    
    return NamedTuple('EvaluationOutputs', [('final_accuracy', float), ('precision', float), ('recall', float), ('f1_score', float)])(
        accuracy, precision, recall, f1
    ) 