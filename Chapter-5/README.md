# Chapter 5: Running pipelines locally

This folder contains the Chapter 5 material for running Kubeflow Pipelines locally (Essential Kubeflow book).

## Contents

- **run_pipeline_locally_executed.ipynb** – Jupyter notebook that runs a pipeline locally (no cluster required). Use this to execute the pipeline step-by-step and inspect inputs/outputs.
- **requirements.txt** – Python dependencies for this chapter.

## Setup

```bash
pip install -r requirements.txt
```

Open the notebook in Jupyter or VS Code:

```bash
jupyter notebook run_pipeline_locally_executed.ipynb
```

## What you’ll do

The notebook walks through running a Kubeflow pipeline on your machine: loading data, training a model, and evaluating it, with artifacts written to local paths. Use it to understand how pipeline components run and how artifacts are produced before moving to a full Kubeflow deployment.

## Notes

- Any local run outputs (e.g. data, metrics, models) are typically written to a local directory; such output folders are gitignored. Re-run the notebook to regenerate them if needed.
