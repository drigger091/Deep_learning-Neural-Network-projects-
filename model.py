from sklearn.model_selection import train_test_split
import data_prep as dp


X , Y = dp.get_XY()

print('Number of Numpy arrays in X(data):',X.shape[0])
print('Height of the arrays in X(data):',X.shape[1])
print('Width of the arrays in X(data):',X.shape[2])
print('Color codes in X(data):',X.shape[3])