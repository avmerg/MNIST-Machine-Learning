#Load relevant SVM packages
from sklearn import svm
from sklearn.svm import SVC

from matplotlib import pyplot as plt
%matplotlib inline 


#Fit the RBF kernel to the data
classifier = svm.SVC(kernel='rbf')
classifier.fit(train_xst, y_train)

#SkLearn documentation referred to

# Make a PCA with different kernels

mnist = tf.keras.datasets.mnist

#Load the data
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#Reshape the data
train_xs = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

# For faster computation, I created a subset of the data
selector = np.random.randint(train_xs.shape[0], size= 3000)
train_xs_subset = train_xs[selector, :]
y_train_subset = y_train[selector]

# Then I made a PCA with two components
model = PCA(n_components=2)
model.fit(train_xs_subset)
train_xs_subset_t = model.transform(train_xs_subset)

# Created a meshgrid with different subsets
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

# Plotted the contours of the decision boundaries
def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

#Designated the SVM Models to be fit to the mesh grid
SVMmodels = (svm.SVC(kernel='linear'),
          svm.SVC(kernel='rbf'),
          svm.SVC(kernel='poly', degree=2),
          svm.SVC(kernel='sigmoid'))

SVMmodels = (clf.fit(train_xs_subset_t, y_train_subset) for clf in SVMmodels)


# Titled the plots
titles = ('SVC (Linear kernel)',
          'SVC (RBF kernel)',
          'SVC (Degree 3 polynomial kernel) ',
          'SVC (Sigmoid kernel)')

# Set-up the grid for plotting
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = train_xs_subset_t[:, 0], train_xs_subset_t[:, 1]
xx, yy = make_meshgrid(X0, X1)

#Create the Graph
for clf, title, ax in zip(SVMmodels, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y_train_subset, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()

#(Plot Template from Sci-kit Learn Developers, 2018) 



