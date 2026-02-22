# Database integration for Kubeflow Pipelines
from kfp import dsl
from kfp.dsl import Dataset, Output


@dsl.component(
    base_image="python:3.11.13",
    packages_to_install=["psycopg2-binary==2.9.9", "pandas==2.0.3"]
)
def fetch_features_from_db(
    feature_query: str,
    output_features: Output[Dataset]
):
    """Fetch features from PostgreSQL database."""
    import psycopg2
    import pandas as pd
    import os

    # Connection details from environment variables
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT', '5432'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )

    # Execute query and load results
    df = pd.read_sql_query(feature_query, conn)
    conn.close()

    # Save to output
    df.to_csv(output_features.path, index=False)
    output_features.metadata['feature_count'] = len(df.columns)
    output_features.metadata['row_count'] = len(df)


@dsl.pipeline(name="Pipeline with Database Features")
def pipeline_with_db():
    fetch_task = fetch_features_from_db(
        feature_query="SELECT * FROM user_features WHERE active = true"
    )

    # Inject database credentials
    from kubernetes import client as k8s_client
    fetch_task.add_env_variable(
        k8s_client.V1EnvVar(
            name='DB_HOST',
            value_from=k8s_client.V1EnvVarSource(
                secret_key_ref=k8s_client.V1SecretKeySelector(
                    name='postgres-credentials',
                    key='DB_HOST'
                )
            )
        )
    )
    # ... repeat for other credential fields
