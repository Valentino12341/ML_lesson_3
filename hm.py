
# BIBLIOTHEKEN IMPORTEREN
# Hiermee kunnen we gegevens verwerken en een AI-model maken

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# DATA INLADEN
# We openen het Titanic-bestand en bekijken de gegevens

data = pd.read_csv("C:/Users/Administrator/Desktop/ML_Ai/lesson_3/titanic2.csv")

print(data.head())
print(data.info())


# ONTBREKENDE WAARDEN INVULLEN
# Lege leeftijden vervangen door de middelste leeftijd (mediaan)
# Lege opstapplaatsen vervangen door de meest voorkomende plaats

data["Age"] = data["Age"].fillna(data["Age"].median())

data["Embarked"] = data["Embarked"].fillna(
    data["Embarked"].mode()[0]
)

print(data.info())


# ONNODIGE KOLOMMEN VERWIJDEREN
# We verwijderen informatie die we niet willen gebruiken

data = data.drop("Cabin", axis=1)
data = data.drop("PassengerId", axis=1)
data = data.drop("Name", axis=1)
data = data.drop("Ticket", axis=1)
data = data.drop("SibSp", axis=1)
data = data.drop("Parch", axis=1)

print(data.head())


# TEKST OMZETTEN NAAR GETALLEN
# Logistic Regression werkt met getallen
# Female = 0 en Male = 1

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

data["Sex"] = encoder.fit_transform(data["Sex"])

data["Embarked"] = encoder.fit_transform(data["Embarked"])

print(data.head())


# X EN Y KIEZEN
# X = gegevens waarmee we het geslacht voorspellen
# Y = het geslacht dat we willen voorspellen

x = data[["Pclass", "Age", "Fare", "Embarked", "Survived"]]

y = data["Sex"]


# DATA VERDELEN
# 80% gebruiken om het model te trainen
# 20% gebruiken om het model te testen

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=2
)


# MODEL MAKEN
# We maken een Logistic Regression model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)


# MODEL TRAINEN
# Het model leert verbanden uit de trainingsgegevens

model.fit(x_train, y_train)


# VOORSPELLINGEN MAKEN
# Het model voorspelt het geslacht van de testpersonen

y_pred = model.predict(x_test)


# NAUWKEURIGHEID BEREKENEN
# We kijken hoeveel voorspellingen goed zijn

from sklearn.metrics import accuracy_score, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)

accuracy = round(accuracy * 100, 2)

print("Nauwkeurigheid =", accuracy, "%")


# CONFUSION MATRIX
# Laat zien hoeveel voorspellingen goed en fout waren

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


# EEN NIEUWE PASSAGIER VOORSPELLEN
# We geven informatie over een passagier
# Pclass = klasse
# Age = leeftijd
# Fare = ticketprijs
# Embarked = opstapplaats (C=0, Q=1, S=2)
# Survived = overleefd (1=ja, 0=nee)

passenger = pd.DataFrame(
    [[1, 25, 80, 2, 1]],
    columns=["Pclass", "Age", "Fare", "Embarked", "Survived"]
)

prediction = model.predict(passenger)


# RESULTAAT TONEN
# 0 betekent vrouw en 1 betekent man

if prediction[0] == 0:
    print("Voorspeld geslacht: Vrouw")
else:
    print("Voorspeld geslacht: Man")
