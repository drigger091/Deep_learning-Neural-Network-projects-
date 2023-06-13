

import numpy as np
from sklearn.model_selection import train_test_split
import data_prep as dp
from model import create_model
import pickle

# Load the data
X, Y = dp.get_XY()

# Split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=2)

# Create and compile the model
model = create_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train,Y_train,validation_split = 0.1,epochs=6)


loss , accuracy = model.evaluate(X_test,Y_test)
print("The Test accuracy =",accuracy*100,"%")

with open('training_history.pkl','wb') as f:
    pickle.dump(history.history,f)

model.save('trained_model.h5')