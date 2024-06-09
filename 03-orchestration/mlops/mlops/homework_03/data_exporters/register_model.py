import mlflow


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-march-2023-experiment")


@data_exporter
def register(data):
    with mlflow.start_run():
        mlflow.set_tag("developer", "agnes")
        mlflow.set_tag("model", "linear regreassion")
        mlflow.log_param("train-data-path", "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet")

        mlflow.log_artifact(local_path="models/lin_reg.bin", artifact_path="models_pickle")
        mlflow.log_model()