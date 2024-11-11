from steps.data_cleaning import clean_data
from steps.evaluation1 import evaluation_model1
from steps.evaluation2 import evaluation_model2
from steps.data_ingest import ingest_data
from steps.model1_dev import Model1
from steps.model2_dev import Model2
from zenml import pipeline

@pipeline(enable_cache=False,)

def Model():

    df=ingest_data()
    X_train1, X_test1, y_train1, y_test1,X_train2, X_test2, y_train2, y_test2=clean_data(df)  
    model1=Model1(X_train1,y_train1)
    rScore1=evaluation_model1(model1,X_test1,y_test1)
    model2=Model2(X_train2,y_train2)
    acc=evaluation_model2(model2,X_test2,y_test2)



    