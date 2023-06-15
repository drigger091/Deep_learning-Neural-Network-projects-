from sklearn.model_selection import train_test_split
from data_prep import get_XY
from model import create_model
import pickle


#load the data

X,Y = get_XY()


#split the data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

model = create_model()

# compile the neural network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
history = model.fit(X_train, Y_train, validation_split=0.1, epochs=7)
model.save('Trained_model.h5')

with open("training_history.pkl",'wb') as f:
    pickle.dump(history.history,f)
loss, accuracy = model.evaluate(X_test, Y_test)
print('Test Accuracy =', accuracy*100,'%')

