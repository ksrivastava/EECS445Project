import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

data = np.genfromtxt('../data/filtered_data.txt', skip_header=True, delimiter='\t')
m = data.shape[1];
n = data.shape[0];

targets = data[:, m - 1]
features = data[:, :(m - 1)]

xTrain = features[:n/2, :]
yTrain = targets[:n/2]

xTest = features[n/2:, :]
yTest = targets[n/2:]

# xTrain_scaled = preprocessing.scale(xTrain)
xTrain_scaled = xTrain

gnb = GaussianNB()
y_pred = gnb.fit(xTrain_scaled, yTrain)
nb_accuracy = gnb.score(xTest, yTest)

print "Naive Bayes: " + str(nb_accuracy * 100)

clf = svm.SVC()
w = clf.fit(xTrain_scaled, yTrain)
cl_accuracy = clf.score(xTest, yTest)

print "Classification: " + str(cl_accuracy * 100)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(xTrain_scaled, yTrain)
nn_accuracy = knn.score(xTest, yTest)

print "NearestNeighbors: " + str(nn_accuracy * 100)