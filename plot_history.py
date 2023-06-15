import pickle
import matplotlib.pyplot as  plt

with open("training_history.pkl",'rb') as f:
    history = pickle.load(f)
    
h = history

# plot the loss value
plt.plot(h.history['loss'], label='train loss')
plt.plot(h.history['val_loss'], label='validation loss')
plt.legend()
plt.savefig("loss_plot.png")
plt.close()

# plot the accuracy value
plt.plot(h.history['acc'], label='train accuracy')
plt.plot(h.history['val_acc'], label='validation accuracy')
plt.legend()
plt.savefig("Accuracy_plot.png")
plt.close()