__author__ = 'yaoliu'


print(__doc__)
import linecache
import matplotlib.pyplot as plt
import csv
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
from sklearn import cross_validation


from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.preprocessing import Imputer


logregClassifier = linear_model.LogisticRegression()
data = np.genfromtxt('../data/SK_manual_kepCatRef_labelled.tsv', skip_header=True, delimiter='\t')

# data = np.genfromtxt('../data/subj.tsv', skip_header=True, delimiter='\t')
# imp = Imputer(missing_values='NaN', strategy='most_frequent', axis = 0)
# data = imp.fit_transform(data)
y = data[:,-1]

n = data.shape[0]
m = data.shape[1]

feature_dir = '../features/features_data/'

################### fsSBMLR
#
# idx_file = 'fsSBMLR_features_idx.txt'
# idx_line = 3
# idx = linecache.getline(feature_dir + idx_file,idx_line).strip().split('\t')
# idx_nu =  map(int, idx)
# idx_nu[:] = [x  for x in idx_nu]
# x = data[:,idx_nu];
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=np.random.RandomState())
#
# y_score = logregClassifier.fit(x_train, y_train).decision_function(x_test)
#
#
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
#
# fpr[0], tpr[0], _ = roc_curve(y_test, y_score)
# roc_auc[0] = auc(fpr[0], tpr[0])
#
#
#
# plt.figure()
# plt.plot(fpr[0], tpr[0], label='ROC curve (area = %0.2f)' % roc_auc[0])
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('fsSBMLR Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()
# # plt.savefig('../features/roc_plots/fsSBMLR')
#
#
#
#
# cv = cross_validation.KFold(n, n_folds=5)
# mean_tpr_cv = 0.0
# mean_fpr_cv = np.linspace(0, 1, 100)
#
# for i, (train, test) in enumerate(cv):
#     probas_ = logregClassifier.fit(x[train], y[train]).predict_proba(x[test])
#     # Compute ROC curve and area the curve
#
#     fpr_cv, tpr_cv, thresholds = roc_curve(y[test], probas_[:, 1])
#     mean_tpr_cv += interp(mean_fpr_cv, fpr_cv, tpr_cv)
#     mean_tpr_cv[0] = 0.0
#     roc_auc_cv = auc(fpr_cv, tpr_cv)
#     plt.plot(fpr_cv, tpr_cv, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc_cv))
#
# plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
#
#
# mean_tpr_cv /= len(cv)
# mean_tpr_cv[-1] = 1.0
# mean_auc_cv = auc(mean_fpr_cv, mean_tpr_cv)
#
# plt.plot(mean_fpr_cv, mean_tpr_cv, 'k--',
#          label='Mean ROC (area = %0.2f)' % mean_auc_cv, lw=2)
#
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic - fsSBMLR')
# plt.legend(loc="lower right")
# plt.savefig('../features/roc_plots/fsSBMLR_cv')
# plt.show()
#
#
#
# ####################  all
#
# # x = data[:,0:m-1]
# # print x.shape[1]
# # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=np.random.RandomState())
# # y_score = logregClassifier.fit(x_train, y_train).decision_function(x_test)
# #
# #
# # fpr = dict()
# # tpr = dict()
# # roc_auc = dict()
# #
# # fpr[0], tpr[0], _ = roc_curve(y_test, y_score)
# # roc_auc[0] = auc(fpr[0], tpr[0])
# #
# #
# #
# # plt.figure()
# # plt.plot(fpr[0], tpr[0], label='ROC curve (area = %0.2f)' % roc_auc[0])
# # plt.plot([0, 1], [0, 1], 'k--')
# # plt.xlim([0.0, 1.0])
# # plt.ylim([0.0, 1.05])
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.title('All features Receiver operating characteristic example')
# # plt.legend(loc="lower right")
# # plt.show()
# #
# #
# # cv = cross_validation.KFold(n, n_folds=5)
# # mean_tpr_cv = 0.0
# # mean_fpr_cv = np.linspace(0, 1, 100)
# #
# # nb = GaussianNB()
# # for i, (train, test) in enumerate(cv):
# #     probas_ = logregClassifier.fit(x[train], y[train]).predict_proba(x[test])
# #     # Compute ROC curve and area the curve
# #
# #     fpr_cv, tpr_cv, thresholds = roc_curve(y[test], probas_[:, 1])
# #     mean_tpr_cv += interp(mean_fpr_cv, fpr_cv, tpr_cv)
# #     mean_tpr_cv[0] = 0.0
# #     roc_auc_cv = auc(fpr_cv, tpr_cv)
# #     plt.plot(fpr_cv, tpr_cv, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc_cv))
# #
# # plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
# #
# #
# # mean_tpr_cv /= len(cv)
# # mean_tpr_cv[-1] = 1.0
# # mean_auc_cv = auc(mean_fpr_cv, mean_tpr_cv)
# #
# # plt.plot(mean_fpr_cv, mean_tpr_cv, 'k--',
# #          label='Mean ROC (area = %0.2f)' % mean_auc_cv, lw=2)
# #
# # plt.xlim([-0.05, 1.05])
# # plt.ylim([-0.05, 1.05])
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.title('Receiver operating characteristic - Base Line')
# # plt.legend(loc="lower right")
# # plt.savefig('../features/roc_plots/sub_base_line_cv_lg')
# # plt.show()
#
# # plt.savefig('../features/roc_plots/all_features')
# #
# # ####################  Half fsFisher
# #
# # idx_file = 'fsFisher_features_idx.txt'
# # idx_line = 3

# # idx = linecache.getline(feature_dir + idx_file,idx_line).strip().split('\t')
# # idx_nu =  map(int, idx)
# # idx_nu[:] = [x for x in idx_nu]
# # print len(idx_nu)
# #
# # half_f = idx_nu[:10]
# # print half_f
# # x = data[:,half_f]
# # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=np.random.RandomState())
# #
# # y_score = logregClassifier.fit(x_train, y_train).decision_function(x_test)
# #
# #
# # fpr = dict()
# # tpr = dict()
# # roc_auc = dict()
# #
# # fpr[0], tpr[0], _ = roc_curve(y_test, y_score)
# # roc_auc[0] = auc(fpr[0], tpr[0])
# #
# #
# #
# # plt.figure()
# # plt.plot(fpr[0], tpr[0], label='ROC curve (area = %0.2f)' % roc_auc[0])
# # plt.plot([0, 1], [0, 1], 'k--')
# # plt.xlim([0.0, 1.0])
# # plt.ylim([0.0, 1.05])
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.title('fsFisher Half features Receiver operating characteristic example')
# # plt.legend(loc="lower right")
# # # plt.savefig('../features/roc_plots/HalfFisher')
# # plt.show()
# #
# #
# # cv = cross_validation.KFold(n, n_folds=5)
# # mean_tpr_cv = 0.0
# # mean_fpr_cv = np.linspace(0, 1, 100)
# #
# # for i, (train, test) in enumerate(cv):
# #     probas_ = logregClassifier.fit(x[train], y[train]).predict_proba(x[test])
# #     # Compute ROC curve and area the curve
# #
# #     fpr_cv, tpr_cv, thresholds = roc_curve(y[test], probas_[:, 1])
# #     mean_tpr_cv += interp(mean_fpr_cv, fpr_cv, tpr_cv)
# #     mean_tpr_cv[0] = 0.0
# #     roc_auc_cv = auc(fpr_cv, tpr_cv)
# #     plt.plot(fpr_cv, tpr_cv, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc_cv))
# #
# # plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
# #
# #
# # mean_tpr_cv /= len(cv)
# # mean_tpr_cv[-1] = 1.0
# # mean_auc_cv = auc(mean_fpr_cv, mean_tpr_cv)
# #
# # plt.plot(mean_fpr_cv, mean_tpr_cv, 'k--',
# #          label='Mean ROC (area = %0.2f)' % mean_auc_cv, lw=2)
# #
# # plt.xlim([-0.05, 1.05])
# # plt.ylim([-0.05, 1.05])
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.title('Receiver operating characteristic - Top 10 Fisher')
# # plt.legend(loc="lower right")
# # plt.savefig('../features/roc_plots/top_10_fisher_cv')
# # plt.show()
# #
#
# #
# # ####################  Half fsInfoGain
# #
# # idx_file = 'fsInfoGain_features_idx.txt'
# # idx_line = 3
# # idx = linecache.getline(feature_dir + idx_file,idx_line).strip().split('\t')
# # idx_nu =  map(int, idx)
# # idx_nu[:] = [x for x in idx_nu]
# #
# #
# # half_f = idx_nu[:10]
# #
# # x = data[:,half_f]
# # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=np.random.RandomState())
# #
# # y_score = logregClassifier.fit(x_train, y_train).decision_function(x_test)
# #
# #
# # fpr = dict()
# # tpr = dict()
# # roc_auc = dict()
# #
# # fpr[0], tpr[0], _ = roc_curve(y_test, y_score)
# # roc_auc[0] = auc(fpr[0], tpr[0])
# #
# #
# #
# # plt.figure()
# # plt.plot(fpr[0], tpr[0], label='ROC curve (area = %0.2f)' % roc_auc[0])
# # plt.plot([0, 1], [0, 1], 'k--')
# # plt.xlim([0.0, 1.0])
# # plt.ylim([0.0, 1.05])
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.title('fsInfoGain Half features Receiver operating characteristic example')
# # plt.legend(loc="lower right")
# # # plt.show()
# # # plt.savefig('../features/roc_plots/HalffsInfoGain')
# #
# #
# # cv = cross_validation.KFold(n, n_folds=5)
# # mean_tpr_cv = 0.0
# # mean_fpr_cv = np.linspace(0, 1, 100)
# #
# # for i, (train, test) in enumerate(cv):
# #     probas_ = logregClassifier.fit(x[train], y[train]).predict_proba(x[test])
# #     # Compute ROC curve and area the curve
# #
# #     fpr_cv, tpr_cv, thresholds = roc_curve(y[test], probas_[:, 1])
# #     mean_tpr_cv += interp(mean_fpr_cv, fpr_cv, tpr_cv)
# #     mean_tpr_cv[0] = 0.0
# #     roc_auc_cv = auc(fpr_cv, tpr_cv)
# #     plt.plot(fpr_cv, tpr_cv, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc_cv))
# #
# # plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
# #
# #
# # mean_tpr_cv /= len(cv)
# # mean_tpr_cv[-1] = 1.0
# # mean_auc_cv = auc(mean_fpr_cv, mean_tpr_cv)
# #
# # plt.plot(mean_fpr_cv, mean_tpr_cv, 'k--',
# #          label='Mean ROC (area = %0.2f)' % mean_auc_cv, lw=2)
# #
# # plt.xlim([-0.05, 1.05])
# # plt.ylim([-0.05, 1.05])
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.title('Receiver operating characteristic - Top 10 InfoGain')
# # plt.legend(loc="lower right")
# # plt.savefig('../features/roc_plots/top_10_infoG_cv')
# # plt.show()
#
#
# #
# # #
# # #
# # #
# # # # nb = SVM()
# # # param_grid = [
# # #   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
# # #   {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
# # #  ]
# # #
# # #
# # # svr = svm.SVC(probability=True)
# # # clf = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, verbose= 1, scoring = roc_auc_score)
# # # prob  =  svr.fit(x_train, y_train).predict_proba(x_test)
# # #
# # # fpr = dict()
# # # tpr = dict()
# # # roc_auc = dict()
# # #
# # # fpr[0], tpr[0], _ = roc_curve(y_test, y_score)
# # # roc_auc[0] = auc(fpr[0], tpr[0])
# # #
# # #
# # # plt.figure()
# # # plt.plot(fpr[0], tpr[0], label='ROC curve (area = %0.2f)' % roc_auc[0])
# # # plt.plot([0, 1], [0, 1], 'k--')
# # # plt.xlim([0.0, 1.0])
# # # plt.ylim([0.0, 1.05])
# # # plt.xlabel('False Positive Rate')
# # # plt.ylabel('True Positive Rate')
# # # plt.title('Receiver operating characteristic example')
# # # plt.legend(loc="lower right")
# # # plt.show()
# # #
# # # nb = GaussianNB()
# # # pro = nb.fit(x_train, y_train).predict_proba(x_test)
# # # fpr[0] ,tpr[0], _ = roc_curve(y_test,  (pro)[:, 1])
# # # roc_auc[0] = auc(fpr[0], tpr[0])
# # #
# # #
# # # plt.figure()
# # # plt.plot(fpr[0], tpr[0], label='ROC curve (area = %0.2f)' % roc_auc[0])
# # # plt.plot([0, 1], [0, 1], 'k--')
# # # plt.xlim([0.0, 1.0])
# # # plt.ylim([0.0, 1.05])
# # # plt.xlabel('False Positive Rate')
# # # plt.ylabel('True Positive Rate')
# # # plt.title('Receiver operating characteristic example')
# # # plt.legend(loc="lower right")
# # # plt.show()
# # #
# # #
# # #
# # #
#

################## fsTest

# idx_file = 'fsTtest_features_idx.txt'
# idx_line = 3
# idx = linecache.getline(feature_dir + idx_file,idx_line).strip().split('\t')
# print idx
# idx_nu =  map(int, idx)
# idx_nu[:] = [x for x in idx_nu]
# half_f = idx_nu[:10]
# x = data[:,half_f];
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=np.random.RandomState())
#
# y_score = logregClassifier.fit(x_train, y_train).decision_function(x_test)
#
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
# plt.title('fsTest Receiver operating characteristic')
# plt.legend(loc="lower right")
# plt.show()
# # plt.savefig('../features/roc_plots/fsSBMLR')
#
#
#
#
# cv = cross_validation.KFold(n, n_folds=5)
# mean_tpr_cv = 0.0
# mean_fpr_cv = np.linspace(0, 1, 100)
#
# for i, (train, test) in enumerate(cv):
#     probas_ = logregClassifier.fit(x[train], y[train]).predict_proba(x[test])
#     # Compute ROC curve and area the curve
#
#     fpr_cv, tpr_cv, thresholds = roc_curve(y[test], probas_[:, 1])
#     mean_tpr_cv += interp(mean_fpr_cv, fpr_cv, tpr_cv)
#     mean_tpr_cv[0] = 0.0
#     roc_auc_cv = auc(fpr_cv, tpr_cv)
#     plt.plot(fpr_cv, tpr_cv, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc_cv))
#
# plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
#
#
# mean_tpr_cv /= len(cv)
# mean_tpr_cv[-1] = 1.0
# mean_auc_cv = auc(mean_fpr_cv, mean_tpr_cv)
#
# plt.plot(mean_fpr_cv, mean_tpr_cv, 'k--',
#          label='Mean ROC (area = %0.2f)' % mean_auc_cv, lw=2)
#
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic - Top 10 T-test')
# plt.legend(loc="lower right")
# plt.savefig('../features/roc_plots/top_10_fsTest_cv')
# # plt.show(
# )

######### Chi Square
#
# idx_file = 'fsChiSquare_features_idx.txt'
# idx_line = 3
# idx = linecache.getline(feature_dir + idx_file,idx_line).strip().split('\t')
# print idx
# idx_nu =  map(int, idx)
# idx_nu[:] = [x for x in idx_nu]
# half_f = idx_nu[:10]
# x = data[:,half_f];
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=np.random.RandomState())
#
# y_score = logregClassifier.fit(x_train, y_train).decision_function(x_test)
#
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
# plt.title('fsTest Receiver operating characteristic')
# plt.legend(loc="lower right")
# plt.show()
#
#
#
#
# cv = cross_validation.KFold(n, n_folds=5)
# mean_tpr_cv = 0.0
# mean_fpr_cv = np.linspace(0, 1, 100)
#
# for i, (train, test) in enumerate(cv):
#     probas_ = logregClassifier.fit(x[train], y[train]).predict_proba(x[test])
#     # Compute ROC curve and area the curve
#
#     fpr_cv, tpr_cv, thresholds = roc_curve(y[test], probas_[:, 1])
#     mean_tpr_cv += interp(mean_fpr_cv, fpr_cv, tpr_cv)
#     mean_tpr_cv[0] = 0.0
#     roc_auc_cv = auc(fpr_cv, tpr_cv)
#     plt.plot(fpr_cv, tpr_cv, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc_cv))
#
# plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
#
#
# mean_tpr_cv /= len(cv)
# mean_tpr_cv[-1] = 1.0
# mean_auc_cv = auc(mean_fpr_cv, mean_tpr_cv)
#
# plt.plot(mean_fpr_cv, mean_tpr_cv, 'k--',
#          label='Mean ROC (area = %0.2f)' % mean_auc_cv, lw=2)
#
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic - Top 10 ChiSquare')
# plt.legend(loc="lower right")
# plt.savefig('../features/roc_plots/top_10_fsChiSquare_cv')
# plt.show()


######### Average Top 10

idx_file = 'ranking_features_list.txt'
idx_line = 1
idx = linecache.getline(feature_dir + idx_file,idx_line).strip().split('\t')
print idx
idx_nu =  map(int, idx)
idx_nu[:] = [x  for x in idx_nu]
half_f = idx_nu[:10]
x = data[:,half_f];

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=np.random.RandomState())

model = logregClassifier.fit(x_train, y_train)
cof = model.coef_[0]


data_file = open('../data/SK_manual_kepCatRef_labelled.tsv', 'rU')
header = next(data_file).strip().split('\t')
data_file.close()

out = open('../features/weight_vectors/top10_weighted_features.txt','wb')

csv_writer = csv.writer(out, delimiter='\t')
for i in range(10):
    record = []
    record.append(idx_nu[i])
    record.append(header[idx_nu[i]])
    record.append(cof[i])
    csv_writer.writerow(record)

out.close()

y_score = model.decision_function(x_test)


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
plt.title('fsTest Receiver operating characteristic')
plt.legend(loc="lower right")
# plt.show()


cv = cross_validation.KFold(n, n_folds=5)
mean_tpr_cv = 0.0
mean_fpr_cv = np.linspace(0, 1, 100)

for i, (train, test) in enumerate(cv):
    probas_ = logregClassifier.fit(x[train], y[train]).predict_proba(x[test])
    # Compute ROC curve and area the curve

    fpr_cv, tpr_cv, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr_cv += interp(mean_fpr_cv, fpr_cv, tpr_cv)
    mean_tpr_cv[0] = 0.0
    roc_auc_cv = auc(fpr_cv, tpr_cv)
    plt.plot(fpr_cv, tpr_cv, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc_cv))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')


mean_tpr_cv /= len(cv)
mean_tpr_cv[-1] = 1.0
mean_auc_cv = auc(mean_fpr_cv, mean_tpr_cv)

plt.plot(mean_fpr_cv, mean_tpr_cv, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc_cv, lw=2)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - Top 10 Average')
plt.legend(loc="lower right")
plt.savefig('../features/roc_plots/top_10_weighted_cv')
# plt.show()


#
# ######### concatenate together
#
# idx_file = 'ranking_features_list_wl2.txt'
# idx_line = 1
# idx = linecache.getline(feature_dir + idx_file,idx_line).strip().split('\t')
# print idx
# idx_nu =  map(int, idx)
# idx_nu[:] = [x for x in idx_nu]
# half_f = idx_nu[:10]
# x = data[:,half_f];
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=np.random.RandomState())
#
# y_score = logregClassifier.fit(x_train, y_train).decision_function(x_test)
#
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
# plt.title('fsTest Receiver operating characteristic')
# plt.legend(loc="lower right")
# plt.show()
#
#
#
#
# cv = cross_validation.KFold(n, n_folds=5)
# mean_tpr_cv = 0.0
# mean_fpr_cv = np.linspace(0, 1, 100)
#
# for i, (train, test) in enumerate(cv):
#     probas_ = logregClassifier.fit(x[train], y[train]).predict_proba(x[test])
#     # Compute ROC curve and area the curve
#
#     fpr_cv, tpr_cv, thresholds = roc_curve(y[test], probas_[:, 1])
#     mean_tpr_cv += interp(mean_fpr_cv, fpr_cv, tpr_cv)
#     mean_tpr_cv[0] = 0.0
#     roc_auc_cv = auc(fpr_cv, tpr_cv)
#     plt.plot(fpr_cv, tpr_cv, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc_cv))
#
# plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
#
#
# mean_tpr_cv /= len(cv)
# mean_tpr_cv[-1] = 1.0
# mean_auc_cv = auc(mean_fpr_cv, mean_tpr_cv)
#
# plt.plot(mean_fpr_cv, mean_tpr_cv, 'k--',
#          label='Mean ROC (area = %0.2f)' % mean_auc_cv, lw=2)
#
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic - Top 10 Average')
# plt.legend(loc="lower right")
# plt.savefig('../features/roc_plots/top_10_avgcv')
# plt.show()