import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
import pybrain.tools.validation

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

data = np.genfromtxt('preprocessed.tsv', delimiter='\t')

targets = data[:, 140]
features = data[:, :140]

print targets
print features

training_set_percentage = 0.7;
test_set_percentage = 1 - training_set_percentage;

#convert categorical features into multiple binary sets

m = features.shape[1]
n = features.shape[0]

training_set_size = int(training_set_percentage * n)
testing_set_size = n - training_set_size

print 'training set size ' + str(training_set_size) + 'testing_set_size' + str(testing_set_size)

xTrain = features[:training_set_size, :]
yTrain = targets[:training_set_size]

xTest = features[training_set_size:, :]
yTest = targets[training_set_size:]

print features.shape


# ds = SupervisedDataSet(140,1)
ds = ClassificationDataSet(140, nb_classes=2, class_labels=['Success','Fail'])

a = 0
# print xTrain[1]
for i in range(targets.shape[0]):
	if targets[i] == -1:
		a +=1
		ds.addSample(features[i],0)
	else:
		ds.addSample(features[i], targets[i])

trndata, tstdata = ds.splitWithProportion( 0.7 )

trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

print 'num of conv' + str(a)
print str(len(ds))
# for inpt, target in ds:
# 	print inpt, target

print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0]

fnn = buildNetwork(trndata.indim,50,trndata.outdim, bias=True, hiddenclass=SoftmaxLayer)
trainer = BackpropTrainer(fnn, dataset=trndata)

for i in range(10):
	print trainer.train()

trnresult = percentError( trainer.testOnClassData(),
                         trndata['class'] )
tstresult = percentError( trainer.testOnClassData(
       dataset=tstdata ), tstdata['class'] )

print "epoch: %4d" % trainer.totalepochs, \
      "  train error: %5.2f%%" % trnresult, \
      "  test error: %5.2f%%" % tstresult

# Feature selection: Univariate


# gnb = GaussianNB()

# parameters = {'kernel':('linear', 'rbf'), 'C':[0.001, 1, 100], 'gamma': [0.001, 0.0001]}
# svr = svm.SVC()
# clf = GridSearchCV(estimator=svr, param_grid=parameters, cv=2)

# parameters = {'n_neighbors':[4, 12, 20], 'algorithm':('auto', 'ball_tree', 'kd_tree', 'brute')}
# kvr = KNeighborsClassifier()
# knn = GridSearchCV(estimator=kvr, param_grid=parameters)


# for i in range(25,26):


#     sel = SelectKBest(chi2, k=i)
#     Train_feat = sel.fit_transform(abs(xTrain), yTrain)
#     Test_feat = sel.transform(xTest)

# features_scores = sel.scores_

# print features_scores
# print features_scores.max()
# print features_scores.argmax()
# print type(features_scores)

#np.savetxt('out.tsv',features_scores.transpose(), delimiter='\t')

# xTrain = LinearSVC(C=1, penalty="l1", dual=False).fit_transform(xTrain,yTrain);

# print xTrain
# print xTrain.shape

# svmselector = svm.SVC(kernel='linear')
# selector = RFE(svmselector, 10, step = 1)
# selector = selector.fit(xTrain, yTrain)
# print "\tTraining accuracy: " + str(selector.score(xTrain, yTrain) * 100)
# print "\tTesting accuracy: " + str(selector.score(xTest, yTest) * 100)
 

# chosen = selector.support_
# print chosen

# for idx, val in enumerate(chosen.tolist()):
#     if str(val) == 'True':
#          print idx

# print selector.ranking_
# # Training


    # gnb.fit(Train_feat, yTrain)
    # clf.fit(Train_feat, yTrain)
    # knn.fit(Train_feat, yTrain)



    # print 'num features' + str(i)

    # # Testing
    # # print "Naive Bayes:"
    # # print "\tTraining accuracy: " + str(gnb.score(Train_feat, yTrain) * 100)
    # # print "\tTesting accuracy: " + str(gnb.score(Test_feat, yTest) * 100)
    # # print confusion_matrix(gnb.predict(Test_feat), yTest)
    # # print "\n"

    # print "Support Vector Classification:"
    # print clf.best_params_
    # print clf.grid_scores_
    # print "\tTraining accuracy: " + str(clf.score(Train_feat, yTrain) * 100)
    # print "\tTesting accuracy: " + str(clf.score(Test_feat, yTest) * 100)
    # print confusion_matrix(clf.predict(Test_feat), yTest)
    # print "\n"

    # # print "Nearest Neighbors:"
    # # print knn.best_params_
    # # print "\tTraining accuracy: " + str(knn.score(Train_feat, yTrain) * 100)
    # # print "\tTesting accuracy: " + str(knn.score(Test_feat, yTest) * 100)
    # # print confusion_matrix(knn.predict(Test_feat), yTest)
    # # print "\n"

