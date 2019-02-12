#Supress warnings because they contain file paths
import warnings
warnings.filterwarnings('ignore')

#Import keras
import tensorflow as tf
import sklearn.decomposition
import pandas as pd
from tensorflow import keras
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

%matplotlib inline 


#Make train and test sets
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


#Reshape data to make it suitable for PCA
print(x_train.shape)
train_xs = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

# Fit the PCA with 2 components
mod = PCA(n_components=2)
mod.fit(train_xs)

# Transform train_xs according to the model
train_xst = mod.transform(train_xs)

# Plot two dimensions, such that the color corresponds to the label
plt.scatter(train_xst[:,0], train_xst[:,1], c = list(y_train))

#Make a PCA with 100 components and fit it to a model
pca = PCA(100)
fullpca = pca.fit(train_xs)

#Plot additional explained variance with every additional component
plt.plot(fullpca.explained_variance_ratio_)
plt.xlabel('Number of components')
plt.ylabel('Additional explained variance')


#Plot cumulative explanation with 100 components
pca = PCA(100)
pca_full = pca.fit(train_xs)

plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')

#Load the 3D Plot 
from mpl_toolkits.mplot3d import Axes3D

#Plot 3 Components 
mod = PCA(n_components=3)
mod.fit(train_xs)

#Transform train_xs appropriately
train_xst = mod.transform(train_xs)

#Plot the principle components
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(train_xst[:,0], train_xst[:,1], train_xst[:,2], cmap=plt.get_cmap('gist_rainbow', 10), c = list(y_train), s = 0.01)
plt.show()

#MatPlot Lib Documentation Referenced 