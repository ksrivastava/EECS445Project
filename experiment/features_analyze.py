__author__ = 'yaoliu'


print(__doc__)
import linecache
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from scipy import interp
from sklearn.metrics import fbeta_score, make_scorer


logregClassifier = linear_model.LogisticRegression()

data = np.genfromtxt('../data/SK_manual_kepCatRef_labelled.tsv', skip_header=True, delimiter='\t')

y = data[:,-1]

n = data.shape[0]
m = data.shape[1]

feature_dir = '../features/features_data/'

################### fsSBMLR

idx_file = 'fsSBMLR_features_idx.txt'
idx_line = 3
idx = linecache.getline(feature_dir + idx_file,idx_line).strip().split('\t')
idx_nu =  map(int, idx)
idx_nu[:] = [x - 1 for x in idx_nu]
x = data[:,idx_nu];

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=np.random.RandomState())

y_score = logregClassifier.fit(x_train, y_train).decision_function(x_test)


fpr = dict()
tpr = dict()
roc_auc = dict()

fpr[0], tpr[0], _ = roc_curve(y_test, y_score)
roc_auc[0] = auc(fpr[0], tpr[0])



plt.figure()
plt.plot(fpr[0], tpr[0], label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('fsSBMLR Receiver operating characteristic example')
plt.legend(loc="lower right")
# plt.show()
plt.savefig('../features/roc_plots/fsSBMLR')

####################  all

x = data[:,0:m-1]
print x.shape[1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=np.random.RandomState())
y_score = logregClassifier.fit(x_train, y_train).decision_function(x_test)


fpr = dict()
tpr = dict()
roc_auc = dict()

fpr[0], tpr[0], _ = roc_curve(y_test, y_score)
roc_auc[0] = auc(fpr[0], tpr[0])



plt.figure()
plt.plot(fpr[0], tpr[0], label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('All features Receiver operating characteristic example')
plt.legend(loc="lower right")
# plt.show()
plt.savefig('../features/roc_plots/all_features')

####################  Half fsFisher

idx_file = 'fsFisher_features_idx.txt'
idx_line = 4
idx = linecache.getline(feature_dir + idx_file,idx_line).strip().split('\t')
idx_nu =  map(int, idx)
idx_nu[:] = [x - 1 for x in idx_nu]


half_f = idx_nu[:len(idx_nu)/2]

x = data[:,half_f]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=np.random.RandomState())

y_score = logregClassifier.fit(x_train, y_train).decision_function(x_test)


fpr = dict()
tpr = dict()
roc_auc = dict()

fpr[0], tpr[0], _ = roc_curve(y_test, y_score)
roc_auc[0] = auc(fpr[0], tpr[0])



plt.figure()
plt.plot(fpr[0], tpr[0], label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('fsFisher Half features Receiver operating characteristic example')
plt.legend(loc="lower right")
# plt.show()
plt.savefig('../features/roc_plots/HalfFisher')

####################  Half fsInfoGain

idx_file = 'fsInfoGain_features_idx.txt'
idx_line = 4
idx = linecache.getline(feature_dir + idx_file,idx_line).strip().split('\t')
idx_nu =  map(int, idx)
idx_nu[:] = [x - 1 for x in idx_nu]


half_f = idx_nu[:len(idx_nu)/2]

x = data[:,half_f]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=np.random.RandomState())

y_score = logregClassifier.fit(x_train, y_train).decision_function(x_test)


fpr = dict()
tpr = dict()
roc_auc = dict()

fpr[0], tpr[0], _ = roc_curve(y_test, y_score)
roc_auc[0] = auc(fpr[0], tpr[0])



plt.figure()
plt.plot(fpr[0], tpr[0], label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('fsInfoGain Half features Receiver operating characteristic example')
plt.legend(loc="lower right")
# plt.show()
plt.savefig('../features/roc_plots/HalffsInfoGain')



#
#
#
# # nb = SVM()
# param_grid = [
#   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#   {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
#  ]
#
#
# svr = svm.SVC(probability=True)
# clf = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, verbose= 1, scoring = roc_auc_score)
# prob  =  svr.fit(x_train, y_train).predict_proba(x_test)
#
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
#
# fpr[0], tpr[0], _ = roc_curve(y_test, y_score)
# roc_auc[0] = auc(fpr[0], tpr[0])
#
#
# plt.figure()
# plt.plot(fpr[0], tpr[0], label='ROC curve (area = %0.2f)' % roc_auc[0])
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()
#
# nb = GaussianNB()
# pro = nb.fit(x_train, y_train).predict_proba(x_test)
# fpr[0] ,tpr[0], _ = roc_curve(y_test,  (pro)[:, 1])
# roc_auc[0] = auc(fpr[0], tpr[0])
#
#
# plt.figure()
# plt.plot(fpr[0], tpr[0], label='ROC curve (area = %0.2f)' % roc_auc[0])
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()
#
#
#
#
