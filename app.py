import pandas as pd
import streamlit as st
import dill
import numpy as np
import json
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
from pipelines.model_pipeline import Model
from zenml.client import Client
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
import pymongo as py


st.set_page_config(page_title= "Predictive Maintainence for Manufacturing Equipment | By M.Kobalan",
                   layout= "wide",
                   initial_sidebar_state= "expanded")

SELECT = option_menu(
    menu_title = None,
    options = ["Home","Model","About Project"],
    default_index=2,
    orientation="horizontal",
    styles={"container": {"padding": "0!important", "background-color": "yellow","size":"cover", "width": "100"},
        "icon": {"color": "black", "font-size": "20px"},   
        "nav-link": {"font-size": "20px", "text-align": "center", "margin": "-2px", "--hover-color": "#6F36AD"},
        "nav-link-selected": {"background-color": "#6F36AD"}})

if SELECT=='Home':
    components.html(f"""<html><p><h1 style="font-family:Neutro; font-size:40px">PROJECT NAME:   PREDICTIVE MAINTAINENCE FOR MANUFACTURING EQUIPMENT</h1></p></html>""")   
    Col1,Col2=st.columns(2)
    with Col1:
        st.image('./Pictures/capture_1.jpg')
        st.image('./Pictures/capture.jpg')
        st.image('./Pictures/capture1.jpg')
        st.image('./Pictures/workflow2.jpg')
        st.image('./Pictures/capture9.jpg')
        st.image('./Pictures/capture10.jpg')
        st.image('./Pictures/capture11.jpg')
        st.image('./Pictures/Predictive-maintenance-workflow.jpg')
    with Col2:
        st.image('./Pictures/capture_2.jpg')
        st.image('./Pictures/maintainence.jpg')
        st.image('./Pictures/capture4.jpg')
        st.image('./Pictures/capture5.jpg')
        st.image('./Pictures/capture6.jpg')
        st.image('./Pictures/capture7.jpg')
        st.image('./Pictures/capture8.jpg')

if SELECT=='About Project':
    Col3,Col4=st.columns(2)
    with Col3:
        components.html("""
                    <html> <body><b>
                    <h1 style="font-family:Google Sans; color:blue;font-size:40px"> About this Project </h1></b>
                    <p style="font-family:Google Sans; font-size:23px">
                        <b>Project_Title</b>: <br>Predictive Maintenance for Manufacturing Equipment <br>
                        <b>Problem Statement:</b>: <br>The goal is to develop a predictive maintenance model that can predict equipment failures before they occur. The dataset includes sensor readings and maintenance logs from a variety of machines.<br>
                        <b>Technologies_Used</b> :<br> Classification, Hyperparameter Tuning, Machine Learning, Product Development, Streamlit <br>
                        <b>Dataset: </b> <a href="https://drive.google.com/file/d/1CgqHeCr132Xj_-UeBlxyx3GlbhkiX8DH/view?usp=drive_link">Link</a><br>
                        <b>Domain </b> : Data Science and MLops <br>
                        <b>Author</b> : M.KOBALAN <br>
                        <b>Linkedin</b> <a href="https://www.linkedin.com/in/kobalan-m-106267227/">Link</a>
                    </p>
                    </body>  </html>  """, height=800,)
    with Col4:
        st.image('./Pictures/overflow.jpg')
        st.image('./Pictures/mlprocess.jpg')

if SELECT=='Model':

    Col5,Col6,Col7=st.columns(3)
    with Col6:
        with st.form('Classification',border=True):
            components.html(f"""<html><p><h1 style="font-family:Google Sans; font-size:40px">MODEL PREDICTION:</h1></p></html>""")   
            Type=st.selectbox(label="Enter the Type ",options=['L','M','H'])	
            Air_temperature=st.text_input(label='Enter the Air Temperature value')
            Process_temperature=st.text_input(label='Enter the Process_temperature value')
            Rotational_speed =st.text_input(label='Enter the Rotational_speed value')
            Torque=st.text_input(label='Enter the Torque value')
            Tool_wear =st.text_input(label='Enter the Tool_wear value')
            button=st.form_submit_button(label='Submit')
            
            if button:       
                    #Model()
                    try:
                        with open('./data/dict_Type.json','r') as File:
                            dict_Type=json.load(File)
                        with open('./data/dict_Fail_Type.json','r') as File:
                            dict_Fail_Type=json.load(File)
                        with open('./artifacts/Scaler.pkl','rb') as File:
                            Scaler=dill.load(File)
                        with open('./artifacts/model1.pkl','rb') as File:
                            model1=dill.load(File)
                        with open('./artifacts/model2.pkl','rb') as File:
                            model2=dill.load(File)
                        columns_value=['Type','Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']
                        datas=[[dict_Type[Type],float(Air_temperature),float(Process_temperature),int(Rotational_speed),float(Torque),int(Tool_wear)]]
                        df=pd.DataFrame(data=datas,columns=columns_value)
                        df[['Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']]=Scaler.transform(df[['Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']])    
                        
                        pred1=model1.predict(df.values)
                        pred2=model2.predict(df.values)
                        key= [key for key, value in dict_Fail_Type.items() if value ==pred2[0]]
                        #print("To inspect your experiment runs within the mlflow UI.You can find your runs tracked within the mlflow_tracker_pipeline experiment")
                        #print(f" For MLFLOW:Now run  mlflow ui --backend-store-uri '{Client().active_stack.experiment_tracker.get_tracking_uri()}'")
                        #print(f" For ZENML:Now run  zenml up --blocking'")
                        
                        st.balloons()                      
                        components.html(f"""<html><body><h1 style="font-family:Google Sans; font-size:25px"> Predicted Target={pred1[0]}<br> Predicted Failure Type={key[0]} </h1></body></html>""")            
                        Client=py.MongoClient('mongodb+srv://kobalanm2705:Kobalan270599@cluster0.ohlri.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
                        mydict = {"Type": Type,'Air temperature [K]': float(Air_temperature),
                                  'Process temperature [K]':float(Process_temperature),'Rotational speed [rpm]':int(Rotational_speed),
                                  'Torque [Nm]':float(Torque),'Tool wear [min]':float(Tool_wear),'Target':int(pred1[0]),'Failure Type':key[0]}
                        db=Client['Project_final']
                        coll=db['data_report']
                        id=coll.insert_one(mydict)
                        print('ID',id.inserted_id)

                        st.success("Model prediction Completed")         
                    except ValueError:
                        st.warning("Please Fill the Value")    
 
        st.image('./Pictures/col_details.jpg')
        
