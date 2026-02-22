#!/usr/bin/env python3
"""
Compile Kubeflow pipeline to YAML.
Usage: python compile_pipeline.py <pipeline_module> <output_path>
"""

import sys
import importlib.util
from pathlib import Path
from kfp import compiler
import kfp.dsl as dsl


def compile_pipeline(pipeline_module_path: str, output_path: str):
    """Compile a pipeline module to YAML."""

    # Load the pipeline module dynamically
    spec = importlib.util.spec_from_file_location("pipeline_module", pipeline_module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load pipeline from {pipeline_module_path}")

    pipeline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline_module)

    # Find the pipeline function
    # Convention: look for a function decorated with @dsl.pipeline
    pipeline_func = None
    for name in dir(pipeline_module):
        obj = getattr(pipeline_module, name)
        if callable(obj) and hasattr(obj, 'pipeline_spec'):
            pipeline_func = obj
            break

    if pipeline_func is None:
        raise ValueError(f"No pipeline function found in {pipeline_module_path}")

    # Compile the pipeline
    compiler.Compiler().compile(
        pipeline_func=pipeline_func,
        package_path=output_path
    )

    print(f"Successfully compiled {pipeline_module_path} to {output_path}")

    # Validation step - make sure the output file exists and isn't empty
    output_file = Path(output_path)
    if not output_file.exists():
        raise FileNotFoundError(f"Compilation produced no output at {output_path}")

    if output_file.stat().st_size == 0:
        raise ValueError(f"Compilation produced empty file at {output_path}")

    print(f"Output file is {output_file.stat().st_size} bytes")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: compile_pipeline.py <pipeline_module> <output_path>")
        sys.exit(1)

    try:
        compile_pipeline(sys.argv[1], sys.argv[2])
    except Exception as e:
        print(f"Compilation failed: {e}")
        sys.exit(1)
