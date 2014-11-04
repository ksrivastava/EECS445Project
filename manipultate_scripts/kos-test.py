import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.cross_validation import KFold

data = np.genfromtxt('../data/filtered_data.txt', skip_header=True, delimiter='\t')
m = data.shape[1] - 1;
n = data.shape[0];

targets = data[:, m]
features = data[:, :m]

training_set_percentage = 0.7;
validation_set_percentage = 0;
test_set_percentage = 1 - training_set_percentage - validation_set_percentage;

training_set_size = int(training_set_percentage * n)
validation_set_size = int(validation_set_percentage * n)
testing_set_size = n - training_set_size - validation_set_size

xTrain = features[:training_set_size, :]
yTrain = targets[:training_set_size]

xVal = features[training_set_size : training_set_size + validation_set_size, :]
yVal = targets[training_set_size : training_set_size + validation_set_size]

xTest = features[training_set_size + validation_set_size:, :]
yTest = targets[training_set_size + validation_set_size:]

# # Feature selection: Univariate
# sel = SelectKBest(chi2, k=7)
# xTrain = sel.fit_transform(abs(xTrain), yTrain)
# xTest = sel.transform(xTest)


gnb = GaussianNB()
clf = svm.SVC()
knn = KNeighborsClassifier(n_neighbors=50)

# loo = KFold(len(yTrain), 6)
# for train_index, test_index in loo: 
# 	t_xTrain = xTrain[train_index]
# 	t_yTrain = yTrain[train_index]
# 	t_xVal = xTrain[test_index]
# 	t_yVal = yTrain[test_index]

# 	gnb.fit(t_xTrain, t_yTrain)
# 	s = (gnb.score(t_xVal, t_yVal))
# 	clf.fit(t_xTrain, t_yTrain)
# 	d = (clf.score(t_xVal, t_yVal))
# 	knn.fit(t_xTrain, t_yTrain)
# 	e = (knn.score(t_xVal, t_yVal))


# xTest = xTrain;
# yTest = yTrain;

# Naive Bayes
y_pred = gnb.fit(xTrain, yTrain)
nb_accuracy = gnb.score(xTest, yTest)
print "Naive Bayes: " + str(nb_accuracy * 100)

# Support Vector Classification
w = clf.fit(xTrain, yTrain)
cl_accuracy = clf.score(xTest, yTest)
print "Classification: " + str(cl_accuracy * 100)

# Nearest Neighbors
knn.fit(xTrain, yTrain)
nn_accuracy = knn.score(xTest, yTest)
print "Nearest Neighbors: " + str(nn_accuracy * 100)