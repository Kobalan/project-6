import logging
import pandas as pd
from zenml import step
import pymongo as py
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
        Client=py.MongoClient('mongodb+srv://kobalanm2705:Kobalan270599@cluster0.ohlri.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
        db=Client['Project_final']
        coll=db['data_report']
        data=[]
        for values in coll.find({},{'_id':0,'UDI':1,'Product ID':1,'Type':1,'Air temperature [K]':1,'Process temperature [K]':1,
                                    'Rotational speed [rpm]':1,'Torque [Nm]':1,'Tool wear [min]':1,'Target':1,'Failure Type':1}):
            
            data.append(values)
        df=pd.DataFrame(data)
        dataset=mlflow.data.from_pandas(df)
        mlflow.log_input(dataset)        
        df.to_csv('./data/Actual_dataset.csv',index=False)
        logging.info(f'Total No. oF Rows {df.shape[0]} and Columns {df.shape[1]}')
        return df
    except Exception as e:
        logging.error(e)
        raise e



