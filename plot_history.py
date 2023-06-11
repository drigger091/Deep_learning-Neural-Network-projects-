import pickle
import matplotlib.pyplot as plt

with open('training_history.pkl','rb') as f:
    history = pickle.load(f)


h = history
#plot the loss
plt.plot(h['loss'],label = 'train loss')
plt.plot(h['val_loss'],label = 'val loss')  
plt.legend()
plt.show()
plt.savefig("Loss_plot.png")
plt.close()


#plot the accuracy value
plt.plot(h['acc'],label = 'train accuracy')
plt.plot(h['val_acc'],label = 'val accuracy')
plt.legend()
plt.show()
plt.savefig("Accuracy_plot.png")
plt.close()
