""" 
Regresion - Classification - Recomendation


        Logistic regresion






"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl

data = pd.read_csv("C:/Users/Administrator/Desktop/ML_Ai/lesson_3/titanic2.csv")
print(data.head())
print(data.info())

data["Age"]=data["Age"].fillna(data["Age"].median(skipna=True),inplace=True)

data["Embarked"]=data["Embarked"].fillna(data["Embarked"].value_counts().idxmax(),inplace=True)

print(data.info())

data = data.drop("Cabin",axis=1)
data = data.drop("PassengerId",axis=1)
data = data.drop("Name",axis=1)
data = data.drop("Ticket",axis=1)
data = data.drop("SibSp",axis=1)
data = data.drop("Parch",axis=1)

print(data.head())
print(data.info())

# LABEL ENCODER 
#turns string in to int  in the colum

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

data["Sex"] = encoder.fit_transform(data["Sex"])

data["Embarked"]= encoder.fit_transform(data["Embarked"])

print(data.head())

x = data[["Pclass","Sex","Age","Fare","Embarked"]]
y = data["Survived"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=2)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x_train,y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix

accuracy = accuracy_score(y_test,y_pred)

print(accuracy)

accuracy = round(accuracy*100,2)

print("Accuracy =",accuracy,"%")