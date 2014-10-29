import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

data = np.genfromtxt('filtered_data.txt', skip_header=True, delimiter='\t')
m = data.shape[1];
n = data.shape[0];

targets = data[:, m - 1]
features = data[:, :(m - 1)]

xTrain = features[:n/2, :]
yTrain = targets[:n/2]

xTest = features[n/2:, :]
yTest = targets[n/2:]

xTrain_scaled = preprocessing.scale(xTrain)

gnb = GaussianNB()
y_pred = gnb.fit(xTrain_scaled, yTrain)
n_a = gnb.score(xTest, yTest)

print n_a*100

clf = svm.SVC()
w = clf.fit(xTrain_scaled, yTrain)
s_a = clf.score(xTest, yTest)

print s_a*100