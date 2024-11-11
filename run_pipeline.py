from pipelines.model_pipeline import Model
from zenml.client import Client
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

if __name__ == "__main__":
    training = Model()

    # training.run()
    print(f"Now run  mlflow ui --backend-store-uri '{Client().active_stack.experiment_tracker.get_tracking_uri()}'")
    print("To inspect your experiment runs within the mlflow UI.You can find your runs tracked within the mlflow_tracker_pipeline experiment")

