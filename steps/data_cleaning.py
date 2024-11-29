import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Union,Tuple,Annotated
from zenml import step
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import dill

class DataStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreprocessStrategy(DataStrategy):
    """
    Data preprocessing strategy which preprocesses the data.
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes columns which are not required, fills missing values with median average values, and converts the data type to float.
        """
        try:
            logging.info("Dropping unneccessary Columns in the DataFrame...")
            data = data.drop(['UDI','Product ID'],axis=1,)
            data=data.drop_duplicates()
            Scaler=MinMaxScaler()       
            numerical_Columns=[ 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
            logging.info("Numerical Column preprocessing Started")
            for column in numerical_Columns:
                if  abs(data[column].skew()) > 0.5:
                    data[column] = np.log(data[column])
                if column in numerical_Columns:
                    Q1 = data[column].quantile(0.25)
                    Q3 = data[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5* IQR
                    upper_bound = Q3 +1.5 * IQR
                    data[column]=data[column].clip(lower_bound,upper_bound)
            Scaler.fit(data[numerical_Columns]) 
            data[numerical_Columns]=Scaler.transform(data[numerical_Columns])  
            with open(r'./artifacts/Scaler.pkl', 'wb') as File:
                dill.dump(Scaler,File)   
            
            logging.info("Categorical Column preprocessing Started...") 
            data['Type']=data['Type'].map({'L':1,'M':2,'H':3})
            data['Failure Type']=data['Failure Type'].map({'No Failure':0,'Power Failure':1,'Tool Wear Failure':2,'Overstrain Failure':3, 'Random Failures':4,'Heat Dissipation Failure':5})
            #data.to_csv('./data/Cleaned_dataset.csv',index=False)
            return data
        except Exception as e:
            logging.error(e)
            raise e


class DataDivideStrategy1(DataStrategy):
    """
    Data dividing strategy which divides the data into train and test data.
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divides the data into train and test data.
        """
        logging.info("DataDivide Strategy1 Started")
        try:
            X = data.drop(['Target', 'Failure Type'], axis=1)
            y = data["Target"]
            logging.info("Over-Sampling technique initiated...")
            SM=SMOTE()
            X_resampled,y_resampled=SM.fit_resample(X,y)
            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled, test_size=0.2, random_state=42
            )
            logging.info("DataDivide Strategy1 Completed...")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error(e)
            raise e
        
class DataDivideStrategy2(DataStrategy):
    """
    Data dividing strategy which divides the data into train and test data.
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divides the data into train and test data.
        """
        logging.info("DataDivide Strategy2 Started...")
        try:
            X = data.drop(['Target', 'Failure Type'], axis=1)
            y = data["Failure Type"]
            logging.info("Over-Sampling technique initiated...")
            SM=SMOTE()
            X_resampled,y_resampled=SM.fit_resample(X,y)
            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled, test_size=0.2, random_state=42
            )
            logging.info("DataDivide Strategy2 Completed...")
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logging.error(e)
            raise e


class DataCleaning:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_data(self.df)


@step
def clean_data(
    data: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "x_train1"],
    Annotated[pd.DataFrame, "x_test1"],
    Annotated[pd.Series, "y_train1"],
    Annotated[pd.Series, "y_test1"],
    Annotated[pd.DataFrame, "x_train2"],
    Annotated[pd.DataFrame, "x_test2"],
    Annotated[pd.Series, "y_train2"],
    Annotated[pd.Series, "y_test2"]
]:

    """Data cleaning class which preprocesses the data and divides it into train and test data.

    Args:
        data: pd.DataFrame
    returns:
        X_train,X_test,y_train,y_test
    """
    try:
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(data, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        divide_strategy1 = DataDivideStrategy1()
        data_cleaning1 = DataCleaning(preprocessed_data, divide_strategy1)
        X_train1, X_test1, y_train1, y_test1=data_cleaning1.handle_data()

        divide_strategy2 = DataDivideStrategy2()
        data_cleaning2 = DataCleaning(preprocessed_data, divide_strategy2)
        X_train2, X_test2, y_train2, y_test2=data_cleaning2.handle_data()
        return X_train1, X_test1, y_train1, y_test1,X_train2, X_test2, y_train2, y_test2
    except Exception as e:
        logging.error(e)
        raise e
