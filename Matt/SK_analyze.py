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
from sklearn.feature_selection import RFECV
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer
from pybrain.utilities           import percentError
from pybrain.structure.modules   import SoftmaxLayer
from sklearn import tree

from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_score

from sklearn import metrics
import matplotlib.pyplot as plt

import sys
# from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
# from scipy import diag, arange, meshgrid, where
# from numpy.random import multivariate_normal
# import pybrain.tools.validation


def roc_curve_start():
    plt.figure()

def add_to_curve(probas, auc, ytrain, ytest, linelabel):

    # print probas.shape
    fpr, tpr, thresholds = metrics.roc_curve(ytest, probas[:,1], pos_label = 1)
    # roc_auc[0] = auc(fpr[0], tpr[0])
    plt.plot(fpr, tpr, label= linelabel + ' (AUC = %0.2f)' % auc)
 
    # plt.savefig('../features/roc_plots/matt_top10')
    # plt.show()

def add_to_curve_c(probas, auc, ytrain, ytest, linelabel):

    # print probas.shape
    fpr, tpr, thresholds = metrics.roc_curve(ytest, probas)
    # roc_auc[0] = auc(fpr[0], tpr[0])
    plt.plot(fpr, tpr, label= linelabel + ' (AUC = %0.2f)' % auc)
 
    # plt.savefig('../features/roc_plots/matt_top10')
    # plt.show()

def finish_curve(title):
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")

    plt.show()

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
        # mat[k] = (v / float(len(actual))) * 100
        mat[k] = v
    
    print "\tPrecision: " + str(mat['TP']/float(mat['TP'] + mat['FP']) * 100)
    print "\tRecall: " + str(mat['TP']/float(mat['TP'] + mat['FN']) * 100)
    print "\tTrue Negative: " + str(mat['TN']/float(mat['TN'] + mat['FP']) * 100)

    return mat

data = np.genfromtxt('preprocessed.tsv', delimiter='\t')

targets = data[:, 138]
features = data[:, :138]

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

#-----------------------------------------------------------------------------
#INITIALIZE MODELS

gnb = GaussianNB()
# , 'C':[0.001, 1, 100]
parameters = {'kernel':('linear', 'rbf'), 'gamma': [0.001, 0.0001], 'class_weight': ['auto'], 'probability': [True]}
svr = svm.SVC()
clf = GridSearchCV(estimator=svr, param_grid=parameters, cv=5)

parameters = {'n_neighbors':[4, 12, 20], 'algorithm':('auto', 'ball_tree', 'kd_tree', 'brute')}
kvr = KNeighborsClassifier()
knn = GridSearchCV(estimator=kvr, param_grid=parameters, cv=5)


parameters = {'criterion':['gini']}
dtc = tree.DecisionTreeClassifier()
dtc = GridSearchCV(estimator=dtc, param_grid=parameters, cv=5)


parameters = {'alpha':[.1, 1, 10]}
rc = linear_model.RidgeClassifier()
rc = GridSearchCV(estimator=rc, param_grid=parameters, cv=5)

parameters = {'loss':['log']}
# sgdclas = linear_model.SGDClassifier(loss="log")
sgdclas = linear_model.SGDClassifier()
sgdclas = GridSearchCV(estimator=sgdclas, param_grid=parameters, cv=5)

parameters = {'penalty':['l2']}
lr = linear_model.LogisticRegression()
lr = GridSearchCV(estimator=lr, param_grid=parameters, cv=5)


parameters = {'C':[.1,1,10]}
pac = linear_model.PassiveAggressiveClassifier()
pac = GridSearchCV(estimator=pac, param_grid=parameters, cv=5)

parameters = {'penalty':['l1']}
per = linear_model.Perceptron()
per = GridSearchCV(estimator=per, param_grid=parameters, cv=5)


parameters = {'n_estimators':[10, 25, 50, 100]}
# ab = AdaBoostClassifier(n_estimators=100)
ab = AdaBoostClassifier()
ab = GridSearchCV(estimator=ab, param_grid=parameters, cv=5)

Train_feat = xTrain
Test_feat = xTest

#-----------------------------------------------------------------------------
# manual feature selection
cfs =[24,  26,  28,  68,  84,  92,  94 , 100 ,102, 104, 121]
fisher = [100 ,102 ,24 , 28,  84,  83,  94,  82,  121, 26 , 93 , 81 , 87 , 42 , 129, 43 , 91,  107, 90 , 80]
manual_list = [27, 30, 81, 84, 97, 107, 112, 113, 119, 130]
# for idx, val in enumerate(manual_list):
#     manual_list[idx] -= 1


# print manual_list
# print xTrain.shape[0]
# Train_feat = np.zeros((1311, 11))
# Test_feat = np.zeros((562, 11))
# for idx, a in enumerate(manual_list):
#     print idx
#     Train_feat[:, idx] = xTrain[:,a]
#     Test_feat[:, idx] = xTest[:,a]

# print type(Train_feat)
# print Train_feat.shape
# print Test_feat.shape
# print Train_feat

# Train_feat = np.zeros((1311, len(cfs)))
# Test_feat = np.zeros((562, len(cfs)))
# for idx, a in enumerate(cfs):
#     print idx
#     Train_feat[:, idx] = xTrain[:,a]
#     Test_feat[:, idx] = xTest[:,a]

# Train_feat = np.zeros((1311, len(fisher)))
# Test_feat = np.zeros((562, len(fisher)))
# for idx, a in enumerate(fisher):
#     print idx
#     Train_feat[:, idx] = xTrain[:,a]
#     Test_feat[:, idx] = xTest[:,a]


    


# sys.exit()
#-----------------------------------------------------------------------------
# chi-squared feature selection

# sel = SelectKBest(chi2, k=20)
# Train_feat = sel.fit_transform(abs(xTrain), yTrain)
# Test_feat = sel.transform(xTest)
# features_scores = sel.scores_
# print sel.scores_
# print sel.get_support()
# print sel.scores_.max()
# print sel.scores_.argmax()
# print type(features_scores)
# b = dict()
# for idx, item in enumerate(features_scores, start=1):
#     b[idx] = item
# for key, value in sorted(b.iteritems(), key=lambda (k,v): (v,k), reverse=True):
#     print "%s: %s" % (key, value)

# np.savetxt('out.tsv',features_scores.transpose(), delimiter='\t')


#-----------------------------------------------------------
# L1 Linear SVM FEATURE SELECTION


# linsvcfs = LinearSVC(C=0.03, penalty="l1", dual=False)
# linsvcfs.fit(xTrain, yTrain)
# print 'coefficients from l1 linear svm selector'
# print linsvcfs.coef_
# print linsvcfs.coef_.shape
# print type(linsvcfs.coef_)

# Train_feat = linsvcfs.transform(xTrain)
# Test_feat = linsvcfs.transform(xTest)

# d = dict()

# for idx, item in enumerate(np.nditer(linsvcfs.coef_), start=1):
#     d[idx] = item
    
# for key, value in sorted(d.iteritems(), key=lambda (k,v): (v,k), reverse=True):
#     print "%s: %s" % (key, value)

#-----------------------------------------------------------
# L1 LOGISTIC REGRESSION - FEATURE SELECTION

# lrfs = linear_model.LogisticRegression(penalty='l1', C=.1)
# lrfs.fit(xTrain, yTrain)
# print 'coefficients from l1 logisitic regression selector'
# print lrfs.coef_

# Train_feat = lrfs.transform(xTrain)
# Test_feat = lrfs.transform(xTest)

# e = dict()

# for idx, item in enumerate(np.nditer(lrfs.coef_), start=1):
#     e[idx] = item
    
# for key, value in sorted(e.iteritems(), key=lambda (k,v): (v,k), reverse=True):
#     print "%s: %s" % (key, value)


#-----------------------------------------------------------
#RECURSIVE FEATURE ELIMINATION  - FEATURE SELECTION USING LINEAR SVM AS MODEL THAT EVALUATES



# svmselector = svm.SVC(kernel='linear')
# selector = RFECV(svmselector, step=1, cv=5)
# selector.fit(xTrain, yTrain)
# print selector.support_
# print selector.ranking_
 
# c= dict()

# for idx, item in enumerate(selector.ranking_, start=1):
#     c[idx] = item


# print 'Rankings using Recursive Feature Elimination\n'    
# for key, value in sorted(c.iteritems(), key=lambda (k,v): (v,k), reverse=True):
#     print "%s: %s" % (key, value)


# print 'number of features selected ' + str(selector.n_features_)
# print selector.support_

# for idx, val in enumerate(selector.support_.tolist()):
#     if str(val) == 'True':
#          print idx

# print selector.ranking_

# Train_feat = selector.transform(xTrain)
# Test_feat = selector.transform(xTest)


#-----------------------------------------------------------


lr.fit(Train_feat, yTrain)

gnb.fit(Train_feat, yTrain)
clf.fit(Train_feat, yTrain)
knn.fit(Train_feat, yTrain)

dtc.fit(Train_feat, yTrain)

rc.fit(Train_feat, yTrain)

sgdclas.fit(Train_feat, yTrain)

pac.fit(Train_feat, yTrain)
per.fit(Train_feat, yTrain)
ab.fit(Train_feat, yTrain)

# print 'num features' + str(i)

# Testing




roc_curve_start()

print "Naive Bayes:"
print "\tTraining accuracy: " + str(gnb.score(Train_feat, yTrain) * 100)
print "\tTesting accuracy: " + str(gnb.score(Test_feat, yTest) * 100)
print confusion_matrix(gnb.predict(Test_feat), yTest)
# print 'auc:'
# print metrics.roc_auc_score(gnb.predict(Test_feat), yTest)
# print metrics.roc_auc_score(yTest, gnb.predict(Test_feat))
# print metrics.roc_auc_score(yTest, gnb.predict_proba(Test_feat)[:,1])
print "\n"
# print gnb.predict(Test_feat).shape
# print gnb.predict_proba(Test_feat).shape
# print gnb.predict_proba(Test_feat)[:,1]
add_to_curve(gnb.predict_proba(Test_feat), metrics.roc_auc_score(yTest, gnb.predict_proba(Test_feat)[:,1], average='weighted'),yTrain, yTest, 'Gaussian Naive Bayes ')

# probas[:,1]


print "Support Vector Classification:"
# print clf.best_params_
# print clf.grid_scores_
print "\tTraining accuracy: " + str(clf.score(Train_feat, yTrain) * 100)
print "\tTesting accuracy: " + str(clf.score(Test_feat, yTest) * 100)
print confusion_matrix(clf.predict(Test_feat), yTest)
# print 'coefficients'
print clf.best_estimator_.coef_
print type(clf.best_estimator_.coef_)
# print 'auc:'

weights = dict()
for idx, item in enumerate(np.nditer(clf.best_estimator_.coef_), start=1):
    print item
    # weights[fisher.index(idx)] = item

# print weights
# print metrics.roc_auc_score(clf.predict(Test_feat), yTest)
# print "\n"
add_to_curve(clf.best_estimator_.predict_proba(Test_feat), metrics.roc_auc_score(yTest, clf.best_estimator_.predict_proba(Test_feat)[:,1], average='weighted'),yTrain, yTest, 'SVM Linear Kernel ')

sys.exit()

print "Nearest Neighbors:"
print knn.best_params_
print "\tTraining accuracy: " + str(knn.score(Train_feat, yTrain) * 100)
print "\tTesting accuracy: " + str(knn.score(Test_feat, yTest) * 100)
print confusion_matrix(knn.predict(Test_feat), yTest)

# print 'auc:'
# print metrics.roc_auc_score(yTest, knn.predict(Test_feat))
# print metrics.roc_auc_score(yTest, knn.best_estimator_.predict(Test_feat))
print "\n"

add_to_curve(knn.best_estimator_.predict_proba(Test_feat), metrics.roc_auc_score(yTest, knn.best_estimator_.predict_proba(Test_feat)[:,1], average='weighted'),yTrain, yTest, 'k-Nearest Neighbors ')
# add_to_curve(knn.best_estimator_.predict_proba(Test_feat), metrics.roc_auc_score(yTest, knn.best_estimator_.predict(Test_feat)),yTrain, yTest, 'k-Nearest Neighbors ')


print "Decision Tree Classifier:"
# print dtc.best_params_
# print dtc.grid_scores_
print "\tTraining accuracy: " + str(dtc.score(Train_feat, yTrain) * 100)
print "\tTesting accuracy: " + str(dtc.score(Test_feat, yTest) * 100)
print confusion_matrix(dtc.predict(Test_feat), yTest)
# print 'auc:'

# print metrics.roc_auc_score(dtc.predict(Test_feat), yTest)
print "\n"
add_to_curve(dtc.predict_proba(Test_feat), metrics.roc_auc_score(yTest, dtc.predict_proba(Test_feat)[:,1], average='weighted'),yTrain, yTest, 'Decision Tree ')

print "Ridge Classifier:"
# print dtc.best_params_
# print dtc.grid_scores_
print "\tTraining accuracy: " + str(rc.score(Train_feat, yTrain) * 100)
print "\tTesting accuracy: " + str(rc.score(Test_feat, yTest) * 100)
print confusion_matrix(rc.predict(Test_feat), yTest)
# print 'auc: first predict first then real first'
# print metrics.roc_auc_score(rc.predict(Test_feat), yTest)
# print metrics.roc_auc_score(yTest, rc.predict(Test_feat))
print "\n"

# print rc.decision_function(Test_feat).shape

# sys.exit()
add_to_curve_c(rc.decision_function(Test_feat), metrics.roc_auc_score(yTest, rc.decision_function(Test_feat), average='weighted'),yTrain, yTest, 'Ridge Regression ')



# print "sgd Classifier:"
# # print dtc.best_params_
# # print dtc.grid_scores_
# print "\tTraining accuracy: " + str(sgdclas.score(Train_feat, yTrain) * 100)
# print "\tTesting accuracy: " + str(sgdclas.score(Test_feat, yTest) * 100)
# print confusion_matrix(sgdclas.predict(Test_feat), yTest)
# print 'auc:'
# print metrics.roc_auc_score(sgdclas.predict(Test_feat), yTest)
# print "\n"
# add_to_curve(sgdclas.predict_proba(Test_feat), metrics.roc_auc_score(yTest, sgdclas.predict_proba(Test_feat)[:,1]),yTrain, yTest, 'SGD based Logistic Regression')

print "log regress Classifier:"
# print dtc.best_params_
# print dtc.grid_scores_
print "\tTraining accuracy: " + str(lr.score(Train_feat, yTrain) * 100)
print "\tTesting accuracy: " + str(lr.score(Test_feat, yTest) * 100)
print confusion_matrix(lr.predict(Test_feat), yTest)
# print 'auc:'
# print metrics.roc_auc_score(lr.predict(Test_feat), yTest)
print "\n"
add_to_curve(lr.predict_proba(Test_feat), metrics.roc_auc_score(yTest, lr.predict_proba(Test_feat)[:,1], average='weighted'),yTrain, yTest, 'L1 Logistic Regression')


# print "PassiveAggressiveClassifier:"
# # print dtc.best_params_
# # print dtc.grid_scores_
# print "\tTraining accuracy: " + str(pac.score(Train_feat, yTrain) * 100)
# print "\tTesting accuracy: " + str(pac.score(Test_feat, yTest) * 100)
# print confusion_matrix(pac.predict(Test_feat), yTest)
# print 'auc:'
# print metrics.roc_auc_score(pac.predict(Test_feat), yTest)
# print "\n"
# add_to_curve_c(pac.decision_function(Test_feat), metrics.roc_auc_score(yTest, pac.decision_function(Test_feat)),yTrain, yTest, 'Passive Aggressive Classifier')

# print "perceptron classifier:"
# # print dtc.best_params_
# # print dtc.grid_scores_
# print "\tTraining accuracy: " + str(per.score(Train_feat, yTrain) * 100)
# print "\tTesting accuracy: " + str(per.score(Test_feat, yTest) * 100)
# print confusion_matrix(per.predict(Test_feat), yTest)
# print 'auc:'
# print metrics.roc_auc_score(per.predict(Test_feat), yTest)
# print "\n"
# add_to_curve_c(per.decision_function(Test_feat), metrics.roc_auc_score(yTest, per.decision_function(Test_feat)),yTrain, yTest, 'Perceptron')



print "adaboost classifier:"

# print dtc.best_params_
# print dtc.grid_scores_
print "\tTraining accuracy: " + str(ab.score(Train_feat, yTrain) * 100)
print "\tTesting accuracy: " + str(ab.score(Test_feat, yTest) * 100)
print confusion_matrix(ab.predict(Test_feat), yTest)


add_to_curve(ab.predict_proba(Test_feat), metrics.roc_auc_score(yTest, ab.predict_proba(Test_feat)[:,1], average='weighted'),yTrain, yTest, 'AdaBoost using Decision Trees')

# scores = cross_val_score(ab, Train_feat, yTrain)
# print scores.mean()
# scores = cross_val_score(ab, Test_feat, yTest)
# print scores.mean()

print "\n"






# ds_train = ClassificationDataSet(Train_feat.shape[1], nb_classes=2, class_labels=['Success','Fail'])
# ds_test = ClassificationDataSet(Train_feat.shape[1], nb_classes=2, class_labels=['Success','Fail'])


# # print xTrain[1]
# for i in range(Train_feat.shape[0]):
#   if yTrain[i] == -1:
#       ds_train.addSample(Train_feat[i],0)
#   else:
#       ds_train.addSample(Train_feat[i], 1)

# for i in range(Test_feat.shape[0]):
#   if yTest[i] == -1:
#       ds_test.addSample(Test_feat[i],0)
#   else:
#       ds_test.addSample(Test_feat[i], 1)


# trndata = ds_train
# tstdata = ds_test
# # trndata, tstdata = ds.splitWithProportion( 0.7 )
# trndata._convertToOneOfMany()
# tstdata._convertToOneOfMany()

# print 'train size: '+ str(len(ds_train))
# print 'test size: '+ str(len(ds_test))
# # for inpt, target in ds:
# #     print inpt, target

# print "Number of training patterns: ", len(trndata)
# print "Input and output dimensions: ", trndata.indim, trndata.outdim
# print "First sample (input, target, class):"
# # print trndata['input'][0], trndata['target'][0], trndata['class'][0]

# fnn = buildNetwork(trndata.indim,50,trndata.outdim, bias=True, hiddenclass=SoftmaxLayer)
# trainer = BackpropTrainer(fnn, dataset=trndata)




# for i in range(25):
#   print trainer.train()

# trnresult = percentError( trainer.testOnClassData(), trndata['class'] )
# tstresult = percentError( trainer.testOnClassData(dataset=tstdata ), tstdata['class'] )


# print "epoch: %4d" % trainer.totalepochs, \
#       "  train error: %5.2f%%" % trnresult, \
#       "  test error: %5.2f%%" % tstresult


# # print type(fnn.activateOnDataset(tstdata))
# # print fnn.activateOnDataset(tstdata).shape


# out =  fnn.activateOnDataset(tstdata)
# maxargs = out.argmax(axis=1)
# # print maxargs.shape
# # print maxargs
# # add_to_curve(out, metrics.roc_auc_score(yTest, maxargs), yTrain, yTest, 'NN')

finish_curve('ROC plot for Top 20 Fisher Test Feature Selection')


