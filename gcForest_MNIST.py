'''
Apply gcForest on MNIST
'''

######################### load packages #######################
import numpy as np
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
from GCForest import gcForest

######################### load datasets #######################
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, y_train = X_train[:2000], y_train[:2000]

######################### reshape #######################
X_train = X_train.reshape((2000, 784))
X_test = X_test.reshape((10000, 784))

######################### build model and train #######################
gcf = gcForest(shape_1X=[28,28], window=[7, 10, 14], tolerance=0.0, min_samples_mgs=10, min_samples_cascade=7)
gcf.fit(X_train, y_train)

######################### predict #######################
y_pred = gcf.predict(X_test)

######################### evaluating accuracy #######################
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print('gcForest accuracy : {}'.format(accuracy))


