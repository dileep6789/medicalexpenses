#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import streamlit as st 
from sklearn.ensemble import GradientBoostingRegressor
from pickle import dump
from pickle import load

st.title('Model Deployment: Prediction of Medical Expenses')

st.sidebar.header('User Input Parameters')

def user_input_features():
    AGE = st.sidebar.number_input("Insert the Age")
    SEX = st.sidebar.selectbox('Gender 1-male 0-Female',('1','0'))
    BMI = st.sidebar.number_input("Insert the BMI:")
    CHILDREN = st.sidebar.selectbox('Childrens',('0','1','2','3','4','5'))
    SMOKER = st.sidebar.selectbox('Smoker 1-yes 0-no',('1','0'))
    REGION = st.sidebar.selectbox('Region 0-southwest 1-southeast 2-northwest 3-northeast',('0','1','2','3'))
    data = {'age':AGE,
            'sex':SEX,
            'bmi':BMI,
            'children':CHILDREN,
            'smoker':SMOKER,
            'region':REGION}
    features = pd.DataFrame(data,index = [0])
    return features 


# In[ ]:


df = user_input_features()
st.subheader('User Input parameters')
st.write(df)


# load the model from disk
loaded_model = load(open('finalized_model.pkl','rb'))

prediction = loaded_model.predict(df)
prediction=np.round(prediction,2)

st.subheader("The Results:")
st.write("The Medical Expenses are:",prediction)
st.success(f"The Medical Expenses are:{prediction[0]}$")
