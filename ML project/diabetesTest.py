import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv('diabetes.csv')

features = ['Pregnancies', 'Glucose','Insulin','BMI','Age']
X = df[features]
Y = df['Outcome']

log = LogisticRegression(max_iter = 200,random_state = 1)
trainX,testX,trainY,testY = train_test_split(X,Y,random_state=0)
log.fit(trainX,trainY)

joblib.dump(log, "diab.pkl")
print("Model saved as house_price_model.pkl")


