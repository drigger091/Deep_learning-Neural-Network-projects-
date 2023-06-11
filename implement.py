from sklearn.model_selection import train_test_split
import data_prep as dp
import tensorflow as tf
from tensorflow import keras
from model import create_model
import pickle


X , Y = dp.get_XY()

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=2)

# scale the data

X_train_scaled = X_train/255
X_test_scaled = X_test/255  

model = create_model()

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])

history = model.fit(X_train_scaled,Y_train,validation_split=0.1,epochs=6)
              
              

loss , accuracy = model.evaluate(X_test_scaled,Y_test)
print("Test Accuracyc=",accuracy*100,"%")


with open("training_history.pkl",'wb') as f:
    pickle.dump(history.history,f)  
