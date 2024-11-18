import logging
import pandas as pd
from zenml import step
from zenml.client import Client
import mlflow
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def ingest_data() -> pd.DataFrame:
    """
    Args:
        None
    Returns:
        df: pd.DataFrame
    """
    try:
        logging.info("Entered the Data Ingestion Component")
        logging.info("Reading the dataset From mongoDB database using DataFrame")
        df=pd.read_csv('./data/Actual_dataset.csv')
        dataset=mlflow.data.from_pandas(df)
        mlflow.log_input(dataset)        
        logging.info(f'Total No. oF Rows {df.shape[0]} and Columns {df.shape[1]}')
        return df
    except Exception as e:
        logging.error(e)
        raise e



