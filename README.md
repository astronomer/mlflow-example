# Example DAGs for Data Science and Machine Learning Use Cases

These examples are meant to be a guide/skaffold for Data Science and Machine Learning pipelines that can be implemented in Airflow with MLflow integration.

In an effort to keep the examples easy to follow, much of the data processing and modeling code has intentionally been kept simple.

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

3. `mlflow-multimodel-config-dag.py` - A simple DS pipeline from data extraction to modeling that leverages the Task Goup API to experiment with multiple models in parallel.
    - This DAG performs the same tasks as example #2 with the addition passing optional grid parameters at runtime to the DAG for various models. 


4. `mlflow-multimodel-register-dag.py` - A simple DS pipeline from data extraction to modeling publication that leverages the Task Goup API to experiment with multiple models in parallel.
    - This DAG performs the same tasks as example #2 with some additions. 
    - Selects the best performing model and paremetesrs then fits a final model on the full dataset for publication to the MLflow Model Registry.
    - Sample runtime configs to pass that will override default parameters provided in `models.py`.

        ```
        {
            "lgbm":{
                "learning_rate": [0.01, 0.05, 0.1], 
                "n_estimators": [50, 100],
                "num_leaves": [31, 40],
                "max_depth": [16, 24, 31]
            },
            "log_reg":{
                "penalty": ["l1","l2","elasticnet"],
                "C": [0.001, 0.01, 0.1, 1, 10],
                "solver": ["newton-cg", "lbfgs", "liblinear"]
            }
        }
        ```


## Sample MLFlow Outputs

### Runs
<img width="1651" alt="image" src="https://user-images.githubusercontent.com/8596749/148460105-e67790d1-9d20-4362-9114-410809ef5b3b.png">


### Plots
<img width="1080" alt="image" src="https://user-images.githubusercontent.com/8596749/148460167-98168739-0667-4bcf-87c3-3351e66cb266.png">

<img width="1092" alt="image" src="https://user-images.githubusercontent.com/8596749/148460387-fc374ae3-4baf-4f4d-8458-ff57a10a6143.png">

### Metrics
<img width="428" alt="image" src="https://user-images.githubusercontent.com/8596749/148460207-bcaa2b02-f0a8-4924-9856-8a5d8d6e14ab.png">

<img width="330" alt="image" src="https://user-images.githubusercontent.com/8596749/148460430-b1cb1577-9735-4506-8c58-7293550bf032.png">

### Parameters

<img width="379" alt="image" src="https://user-images.githubusercontent.com/8596749/148460277-f76b444a-f1ef-40da-ac20-d2aef7c9e59a.png">

