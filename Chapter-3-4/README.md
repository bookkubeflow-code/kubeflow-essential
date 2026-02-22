# Chapter 3–4: Kubeflow Pipelines essentials

This folder contains pipelines and components for the Kubeflow Pipelines (Chapters 3–4) material from the Essential Kubeflow book.

## Contents

### Pipelines

- **ml_training_pipeline.py** – ML training pipeline (load data → train → evaluate) on the Iris dataset. Compiles to `ml_pipeline.yaml`.
- **smart_caching_pipeline.py** – Pipeline demonstrating caching and reuse of pipeline outputs.
- **advanced_parameter_pipeline.py** – Pipeline with parameters and configuration.
- **advanced_storage_pipeline.py** – Storage and artifact handling patterns.
- **advanced_error_handling_pipeline.py** – Error handling, retries, and fallbacks.

### Components

- **components/ml_ops_components.py** – Reusable components: `load_data`, `train_model`, `evaluate_model`.

### Configuration and running

- **config.py** – KFP connection and pipeline root configuration (local vs cloud).
- **pipeline_runner.py** – Run and monitor pipelines (with optional Dex auth).
- **run_analyzer.py** – Analyze and inspect pipeline runs.

### Artifacts

- **ml_pipeline.yaml** – Compiled pipeline definition (generated from `ml_training_pipeline.py`).

## Setup

```bash
pip install -r requirements.txt
```

Set `KFP_LOCAL_MODE=true` and optionally `PIPELINE_ROOT` for local runs. Use the book’s instructions for Kubeflow/Dex if running against a cluster.

## Running the ML pipeline

```bash
python ml_training_pipeline.py   # compile to ml_pipeline.yaml
# Then run via pipeline_runner.py or upload ml_pipeline.yaml to the KFP UI.
```

## Notes

- Pipeline outputs (e.g. from local runs) may appear under `local_outputs/`; this folder is gitignored.
- Do not commit `__pycache__/` or `local_outputs/`.
