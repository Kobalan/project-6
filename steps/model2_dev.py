import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin
from zenml import step
from zenml.client import Client
import mlflow
import dill
experiment_tracker = Client().active_stack.experiment_tracker

class Model_Trainer:
    """
    Model that implements the Model interface.
    """
    def __init__(self):
        pass

    def train(self, x_train, y_train):
        """
        Trains the model on the given data.

        Args:
            x_train: Training data
            y_train: Target data
        """
        reg = RandomForestClassifier(max_depth=8, max_features= 'log2', max_leaf_nodes=9, min_samples_leaf=4, min_samples_split= 5, n_estimators= 100)
        reg.fit(x_train, y_train)
        return reg


@step(experiment_tracker=experiment_tracker.name)

def Model2(
    x_train: pd.DataFrame,
    y_train: pd.Series,
)-> ClassifierMixin:
    """
    Args:
        x_train: pd.DataFrame
        y_train: pd.Series
    Returns:
        model: ClassifierMixin
    """
    try:
        logging.info("Model2 Training Started..")
        mlflow.sklearn.autolog()
        model = Model_Trainer()
        rf_Model=model.train(x_train.values, y_train)
        logging.info("Saving the trained Model")
        with open(r'./saved_models/model2.pkl', 'wb') as File:
                dill.dump(rf_Model,File)
        return rf_Model
        
    except Exception as e:
        logging.error(e)
        raise e
