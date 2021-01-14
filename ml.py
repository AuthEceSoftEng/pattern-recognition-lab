import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix

def load_data():

  iris = datasets.load_iris()

  return iris.data, iris.target, iris["feature_names"]


def plot_data(training, target, feature_names):
  plt.scatter(training[:, 0], training[:, 2], c=target)
  plt.xlabel(feature_names[0])
  plt.ylabel(feature_names[2])
  plt.show()

def perform_clustering(training):

  estimators = [('k_means_iris_3', KMeans(n_clusters=3))]
  titles = ['3 clusters']

  fignum = 1
  for name, est in estimators:
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(training)
    labels = est.labels_

    ax.scatter(training[:, 3], training[:, 0], training[:, 2], c=labels.astype(float), edgecolor='k')

    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1

  plt.show()

def perform_classification(training, target):

  clf = svm.SVC(kernel="rbf", gamma=0.01)
  clf.fit(training, target)

  pred = clf.predict(training)

  print("Accuracy:", accuracy_score(target, pred))
  print("Confusion Matrix:\n", confusion_matrix(target, pred))

# Get data
training, target, feature_names = load_data()

# Plot data
# plot_data(training, target, feature_names)

# Perform Classification
# perform_classification(training, target)

# Perform Clustering
# perform_clustering(training)
