import pandas as pd
import numpy as np
import streamlit as st 
from sklearn.tree import  DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pickle import dump
import pickle 
from pickle import load

st.title('Model Deployment: LOAN APPROVAL')

st.sidebar.header('User Input Parameters')

def user_input_features():
    Education = st.selectbox('Education',("1","0"))
    Married = st.selectbox('Marital Status',("0","1")) 
    Credit_History = st.selectbox('Credit_History',("0","1"))
    Property_Area = st.selectbox('Property_Area',("0","1","2"))
    data = {'Married':Married,'Education':Education,'Credit_History':Credit_History,'Property_Area':Property_Area}
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)


data_set = pd.read_csv("train.csv")
data_set = data_set.drop(['Loan_ID','Dependents'],axis=1)
data = data_set.copy()
##Filling null values with mode
data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
data['Married'].fillna(data['Married'].mode()[0], inplace=True)
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)
data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)
data['LoanAmount'].fillna(data['LoanAmount'].mean(), inplace=True)
data['Loan_Status'] = data['Loan_Status'].map({'Y':1, 'N':0})

label_encode = LabelEncoder()
# Encode labels in column 'Gender'
data["Gender"] = label_encode.fit_transform(data["Gender"])
# Encode labels in column 'Married'
data["Married"] = label_encode.fit_transform(data["Married"])
# Encode labels in column 'Education'
data["Education"] = label_encode.fit_transform(data["Education"])
# Encode labels in column 'Property-area'
data["Property_Area"] = label_encode.fit_transform(data["Property_Area"])
# Encode labels in column 'self-employed'
data["Self_Employed"] = label_encode.fit_transform(data["Self_Employed"])

x = np.array(data.iloc[:,[1,2,8,9]])
y = np.array(data["Loan_Status"])

scaler = StandardScaler()
X = scaler.fit_transform(x)



x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)


d_tree = DecisionTreeClassifier(criterion="entropy" , max_depth=3)
d_tree.fit(x_train,y_train)
tree_pred = d_tree.predict(x_test)





prediction = d_tree.predict(df)
prediction_proba = d_tree.predict_proba(df)


st.subheader('Predicted Result')
st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')

st.subheader('Prediction Probability')
st.write(prediction_proba)
