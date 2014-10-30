import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

data = np.genfromtxt('../data/filtered_data.txt', skip_header=True, delimiter='\t')
m = data.shape[1];
n = data.shape[0];

training_set_size = int(3.0/4.0 * n)
testing_set_size = n - training_set_size

targets = data[:, m - 1]
features = data[:, :(m - 1)]

xTrain = features[:training_set_size, :]
yTrain = targets[:training_set_size]

xTest = features[training_set_size:, :]
yTest = targets[training_set_size:]

# Feature selection: Univariate
sel = SelectKBest(chi2, k=7)
xTrain = sel.fit_transform(abs(xTrain), yTrain)
xTest = sel.transform(xTest)

# Naive Bayes
gnb = GaussianNB()
y_pred = gnb.fit(xTrain, yTrain)
nb_accuracy = gnb.score(xTest, yTest)
print "Naive Bayes: " + str(nb_accuracy * 100)

# Support Vector Classification
clf = svm.SVC()
w = clf.fit(xTrain, yTrain)
cl_accuracy = clf.score(xTest, yTest)
print "Classification: " + str(cl_accuracy * 100)

# Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(xTrain, yTrain)
nn_accuracy = knn.score(xTest, yTest)
print "Nearest Neighbors: " + str(nn_accuracy * 100)