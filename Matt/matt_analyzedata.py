import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.grid_search import GridSearchCV

data = np.genfromtxt('matt_data.tsv', skip_header=True, delimiter='\t')
m = data.shape[1] - 1;
n = data.shape[0];


#fill in missing values using median
imp = Imputer(missing_values=-6, strategy='median', axis = 0)
complete = imp.fit_transform(data) 


targets = complete[:, 0]
#features = complete[:, 1:]

# print 'targets'
# print targets

# print 'features'
# print features


training_set_percentage = 0.7;
test_set_percentage = 1 - training_set_percentage;

#convert categorical features into multiple binary sets
enc = OneHotEncoder()
features = enc.fit_transform(complete[:, 1:]).toarray()

m = features.shape[1]
n = features.shape[0]

print 'dimensions ' + str(m) + ' by ' + str(n) 

#np.savetxt('missingfilledin.txt', complete, delimiter='\t')

#features = transformed.toarray()

training_set_size = int(training_set_percentage * n)
testing_set_size = n - training_set_size

xTrain = features[:training_set_size, :]
yTrain = targets[:training_set_size]

xTest = features[training_set_size:, :]
yTest = targets[training_set_size:]

# Switch successful and unsuccessful
yTrain *= -1
yTest *= -1

# Feature selection: Univariate
sel = SelectKBest(chi2, k=15)
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


def confusion_matrix(pred, actual):
    mat = {'TP' : 0, 'FP' : 0, 'TN' : 0, 'FN' : 0}
    for i in range(0, len(actual)):
        if actual[i] == 1:
            if pred[i] == 1:
                mat['TP'] += 1
            else:
                mat['FP'] += 1
        else:
            if pred[i] == -1:
                mat['TN'] += 1
            else:
                mat['FN'] += 1

    for k, v in mat.items():
        mat[k] = (v / float(len(actual))) * 100

    return mat

# Testing
print "Naive Bayes:"
print "\tTraining accuracy: " + str(gnb.score(xTrain, yTrain) * 100)
print "\tTesting accuracy: " + str(gnb.score(xTest, yTest) * 100)
print confusion_matrix(gnb.predict(xTest), yTest)
print "\n"

print "Support Vector Classification:"
print clf.best_params_
print "\tTraining accuracy: " + str(clf.score(xTrain, yTrain) * 100)
print "\tTesting accuracy: " + str(clf.score(xTest, yTest) * 100)
print confusion_matrix(clf.predict(xTest), yTest)
print "\n"

print "Nearest Neighbors:"
print knn.best_params_
print "\tTraining accuracy: " + str(knn.score(xTrain, yTrain) * 100)
print "\tTesting accuracy: " + str(knn.score(xTest, yTest) * 100)
print confusion_matrix(knn.predict(xTest), yTest)
print "\n"
