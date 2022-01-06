# Example DAGs for Data Science and Machine Learning Use Cases

These examples are meant to be a guide/skaffold for Data Science and Machine Learning pipelines that can be implemented in Airflow with MLflow integration.

In an effort to keep the examples easy to follow, much of the data processing and modeling code has intentially been kept simple.

## Examples

1. `mlflow-dag.py` - A simple DS pipeline from data extraction to modeling.
    - Pulls data from BigQuery using the Google Provider (BigQueryHook) into a dataframe that preps, trains, and builds the model
    - Data is passed between the tasks using [XComs](https://airflow.apache.org/docs/apache-airflow/stable/concepts/xcoms.html)
    - Uses GCS as an Xcom backend to easily track intermediary data in a scalable, external system
    - Trains model with Grid Search
    - Logs model metrics to MLflow.

2. `mlflow-multimodel-dag.py` - A simple DS pipeline from data extraction to modeling that leverages the Task Goup API to experiment with multiple models in parallel.
    - This DAG performs the same tasks as example #1 with some additions. 
    - Uses Task Groups to configure training multiple models with Grid Search in parallel.



## Sample MLFlow Outputs

### Runs


### Plots


### Metrics


