import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.grid_search import GridSearchCV

data = np.genfromtxt('../data/filtered_data.txt', skip_header=True, delimiter='\t')
m = data.shape[1] - 1;
n = data.shape[0];

targets = data[:, m]
features = data[:, :m]

training_set_percentage = 0.7;
test_set_percentage = 1 - training_set_percentage;

training_set_size = int(training_set_percentage * n)
testing_set_size = n - training_set_size

xTrain = features[:training_set_size, :]
yTrain = targets[:training_set_size]

xTest = features[training_set_size:, :]
yTest = targets[training_set_size:]

# Feature selection: Univariate
sel = SelectKBest(chi2, k=10)
xTrain = sel.fit_transform(abs(xTrain), yTrain)
xTest = sel.transform(xTest)


gnb = GaussianNB()

parameters = {'kernel':('linear', 'rbf'), 'C':[0.001, 1, 100], 'gamma': [0.001, 0.0001]}
svr = svm.SVC()
clf = GridSearchCV(estimator=svr, param_grid=parameters, cv=2)

parameters = {'n_neighbors':[4, 12, 20], 'algorithm':('auto', 'ball_tree', 'kd_tree', 'brute')}
kvr = KNeighborsClassifier()
knn = GridSearchCV(estimator=kvr, param_grid=parameters)

# Training
gnb.fit(xTrain, yTrain)
clf.fit(xTrain, yTrain)
knn.fit(xTrain, yTrain)


# Testing
print "Naive Bayes:"
print "\tTraining accuracy: " + str(gnb.score(xTrain, yTrain) * 100)
print "\tTesting accuracy: " + str(gnb.score(xTest, yTest) * 100)
print "\n"

print "Support Vector Classification:"
print "\tTraining accuracy: " + str(clf.score(xTrain, yTrain) * 100)
print "\tTesting accuracy: " + str(clf.score(xTest, yTest) * 100)
print "\n"

print "Nearest Neighbors:"
print "\tTraining accuracy: " + str(knn.score(xTrain, yTrain) * 100)
print "\tTesting accuracy: " + str(knn.score(xTest, yTest) * 100)
print "\n"
