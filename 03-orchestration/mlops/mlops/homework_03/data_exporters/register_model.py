import mlflow

@data_exporter
def register(data):
    mlflow.set_tracking_uri("sqlite:///home/mlflow/mlflow.db")
    mlflow.set_experiment("nyc-taxi-march-2023-experiment")
    with mlflow.start_run():
        mlflow.set_tag("developer", "agnes")
        mlflow.set_tag("model", "linear regreassion")
        mlflow.log_param("train-data-path", "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet")

        mlflow.sklearn.log_model(data[0], artifact_path="models")
        mlflow.log_artifact(local_path="mlops/homework_03/models/dv.b", artifact_path="vectorizer")
        print("Completed run")