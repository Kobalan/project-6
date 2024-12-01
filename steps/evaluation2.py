import logging
from abc import ABC, abstractmethod
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,f1_score,recall_score
import logging
import numpy as np
import pandas as pd
import mlflow
from sklearn.base import ClassifierMixin
#from typing_extensions import Annotated
from zenml import step
from zenml.client import Client
#from typing import Tuple

#experiment_tracker = Client().active_stack.experiment_tracker

class Evaluation(ABC):
    """
    Abstract Class defining the strategy for evaluating model performance
    """
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass


class ROC_AUC_Score(Evaluation):
    """
    Evaluation strategy that uses ROC_AUC_Score
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            mse: float
        """
        try:
            logging.info("Entered the calculate_score method of the ROC_AUC_Score class...")
            rScore = roc_auc_score(y_true, y_pred,multi_class='ovr')
            logging.info("The ROC_AUC_Score value is: " + str(rScore))
            return rScore
        
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the ROC_AUC_Score class. Exception message:  "
                + str(e)
            )
            raise e


class Accuracy_Score(Evaluation):
    """
    Evaluation strategy that uses Accuracy Score
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            r2_score: float
        """
        try:
            logging.info("Entered the calculate_score method of the Accuracy_Score class...")
            acc = accuracy_score(y_true, y_pred)
            logging.info("The Accuracy score value is: " + str(acc))
            return acc
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the Accuracy_Score class. Exception message:  "
                + str(e)
            )
            raise e


class Precision_Score(Evaluation):
    """
    Evaluation strategy that uses Precision Score
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            r2_score: float
        """
        try:
            logging.info("Entered the calculate_score method of the Precision_Score class...")
            Prec = precision_score(y_true, y_pred,average='weighted')
            logging.info("The Precision score value is: " + str(Prec))
            return Prec
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the Precision_Score class. Exception message:  "
                + str(e)
            )
            raise e

class F1_Score(Evaluation):
    """
    Evaluation strategy that uses F1 Score
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            r2_score: float
        """
        try:
            logging.info("Entered the calculate_score method of the F1_Score class...")
            F1 = f1_score(y_true, y_pred,average='weighted')
            logging.info("The F1 score value is: " + str(F1))
            return F1
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the F1_Score class. Exception message:  "
                + str(e)
            )
            raise e

class recall_Score(Evaluation):
    """
    Evaluation strategy that uses recall Score
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            r2_score: float
        """
        try:
            logging.info("Entered the calculate_score method of the recall_Score class...")
            recall = recall_score(y_true, y_pred,average='weighted')
            logging.info("The recall score value is: " + str(recall))
            return recall
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the recall_Score class. Exception message:  "
                + str(e)
            )
            raise e


@step#(experiment_tracker=experiment_tracker.name)
def evaluation_model2(
    model: ClassifierMixin, x_test: pd.DataFrame, y_test: pd.Series
): 
#->Tuple[Annotated[float, "Prec_score"], Annotated[float, "ACC_score"]]:

    """
    Args:
        model: ClassifierMixin
        x_test: pd.DataFrame
        y_test: pd.Series
    Returns:
        roc_auc_score: float
        accuracy: float
    """
    try:

        logging.info("Evaluation2 Started...")
        
        prediction_proba=model.predict_proba(x_test.values)
        rScore_class = ROC_AUC_Score()
        roc_auc =rScore_class.calculate_score(y_test, prediction_proba)

        prediction = model.predict(x_test.values)
        Acc = Accuracy_Score()
        ACC_score = Acc.calculate_score(y_test, prediction)        
        
        prec = Precision_Score()
        Prec_score = prec.calculate_score(y_test, prediction)     

        F1 = F1_Score()
        F1_score = F1.calculate_score(y_test, prediction)

        recall = recall_Score()
        Recall_Score = recall.calculate_score(y_test, prediction)

        with mlflow.start_run(run_name="Multi_Class_Evaluation",nested=True):
            mlflow.log_metric("test_Accuracy_score", ACC_score)
            mlflow.log_metric("test_Recall_Score", Recall_Score)
            mlflow.log_metric("test_F1_score", F1_score)
            mlflow.log_metric("test_Precision_score", Prec_score)
            mlflow.log_metric("roc_auc_Score", roc_auc)
        return ACC_score
    
    except Exception as e:
        logging.error(e)
        raise e
