# Importing Python libraries
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Data Preprocessing
dataset = pd.read_csv("Churn_Modelling.csv")
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(x)
print(y)
# Labelling Encoding
label_encoder = LabelEncoder()
x[:, 2] = label_encoder.fit_transform(x[:, 2])
print(x)
# Geography Column
column_transformer = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough"
)
x = np.array(column_transformer.fit_transform(x))
print(x)
# Spliting data set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# Feature Scaling
standard_scaler = StandardScaler()
x_train = standard_scaler.fit_transform(x_train)
x_test = standard_scaler.transform(x_test)
# Including ANN
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation="swish"))
ann.add(tf.keras.layers.Dense(units=6, activation="swish"))
ann.add(tf.keras.layers.Dense(units=1, activation="tanh"))
# Training ANN
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
ann.fit(x_train, y_train, batch_size=32, epochs=20)
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
# ann.compile(optimizer = 'adam' , loss = 'binary_crossentropy', metrics =['accuracy'])
# ann.fit(x_train,y_train,batch_size = 32, epochs = 20)
# Predicting and evaluating models
print(
    ann.predict(
        standard_scaler.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
    )
)
# Predicting Test Results
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
with open("classifier.dat", "wb") as out_file:
    pickle.dump(classifier, out_file)
    print("Saved Successfully")
ann.save("./model.h5")
print(
    np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
