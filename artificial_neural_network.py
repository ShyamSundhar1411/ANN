import tensorflow as t
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
d = pd.read_csv('Churn_Modelling.csv')
x = d.iloc[:, 3:-1].values
y = d.iloc[:,-1].values
print(x)
print(y)
r = LabelEncoder()
x[:,2]=r.fit_transform(x[:,2])
print(x)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
c = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(c.fit_transform(x))
print(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
p = StandardScaler()
x_train=p.fit_transform(x_train)
x_test-p.transform(x_test)
ann = t.keras.models.Sequential()
ann.add(t.keras.layers.Dense(units = 6,activation = 'relu'))
ann.add(t.keras.layers.Dense(units = 6,activation='relu'))
ann.add(t.keras.layers.Dense(units = 1,activation='sigmoid'))
ann.compile(optimizer = 'adam' , loss = 'binary_crossentropy', metrics =['accuracy'])
ann.fit(x_train,y_train,batch_size = 32, epochs = 20)
print(ann.predict(p.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]])))
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
