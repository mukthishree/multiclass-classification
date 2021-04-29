import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense,LeakyReLU,PReLU,ELU,Dropout,Activation
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

dataset = pd.read_csv('data.csv')
dataset = dataset.rename(columns={'diagnosis':'label'})

y = dataset["label"].values
labelencoder = LabelEncoder()
Y =  labelencoder.fit_transform(y)
X = dataset.iloc[:, 2:31]
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)

model = Sequential()
model.add(Dense(16,activation='relu',input_dim = 29))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])
history = model.fit(X_train, y_train,validation_split=0.33, batch_size = 10, verbose = 1,epochs = 100, validation_data = (X_test, y_test))

loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1,len(loss)+1)
plt.plot(epochs,loss,"y",label="Train_loss")
plt.plot(epochs,val_loss,"r",label="Val_loss")
plt.title("Training and Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
epochs = range(1,len(loss)+1)
plt.plot(epochs,acc,"y",label="Train_acc")
plt.plot(epochs,val_acc,"r",label="Val_acc")
plt.title("Training and Validation Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

'''from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)'''

from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)
print("score:", score)




