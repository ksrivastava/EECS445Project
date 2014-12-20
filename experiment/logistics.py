__author__ = 'yaoliu'

print(__doc__)
import linecache
import os
import fnmatch
import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
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
from sklearn import linear_model

data = np.genfromtxt('../data/preprocessed_with_one_hot.tsv', skip_header=True, delimiter='\t')

n = data.shape[0]
m = data.shape[1]
x = data[:,:m-1]
y = data[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)
print x_train.shape
n_train = x_train.shape[0]
print n_train

######################### Add more features file under this folder ######################

feature_dir = '../features/features_data/'
idx_line = 3
##########################################################################


######################### Feature index should be in line 3 ######################

field = ['feature_selection_algorithm', 'classifier']
##########################################################################

######################### Add more number if you want to have more range of top features listed  ######################

top_features_list = [1, 3, 5, 8, 10, 20]

##########################################################################

top_features_list_str = map(str, top_features_list)
header = field + top_features_list_str
mean_auc_file = open(feature_dir + 'auc_mean.tsv','wb')
writer = csv.writer(mean_auc_file, delimiter='\t')
writer.writerow(header)


cls_dic = dict()
######################### Add more classifiers here ######################
cls_dic['LogReg'] = linear_model.LogisticRegression()
#############################################################################


i = 0
for file in os.listdir(feature_dir):
    if fnmatch.fnmatch(file, '*_idx.txt'):
        idx = linecache.getline(feature_dir + file, idx_line).strip().split('\t')
        print idx
        idx_nu = map(int, idx)
        algorithm = file.split('_')[0]

        for key, value in cls_dic.iteritems():

            record = []
            record.append(algorithm)
            record.append(key)
            classifier = value
            for j in top_features_list:
                top_f = idx_nu[:j]
                print top_f
                x_top_f_train = x_train[:,top_f]
                x_top_f_test = x_test[:,top_f]
                print x.shape
                cv = KFold(n_train, n_folds=5)

                mean_tpr_cv = 0.0
                mean_fpr_cv = np.linspace(0, 1, 100)

                for i, (train, test) in enumerate(cv):
                     probas_ = classifier.fit(x_top_f_train[train], y_train[train]).predict_proba(x_top_f_train[test])
                     fpr_cv, tpr_cv, thresholds = roc_curve(y_train[test], probas_[:, 1])
                     mean_tpr_cv += interp(mean_fpr_cv, fpr_cv, tpr_cv)
                     roc_auc_cv = auc(fpr_cv, tpr_cv)

                # plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
                mean_tpr_cv /= len(cv)
                mean_tpr_cv[-1] = 1.0
                mean_auc_cv = auc(mean_fpr_cv, mean_tpr_cv)
                print mean_auc_cv
                record.append(mean_auc_cv)
            writer.writerow(record)


            ###################### Draw top 10 features ROC curve on testing set ####################
            ###### change which pot to draw if necessary
            ##########################################################################
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            probas_ = classifier.fit(x_top_f_train, y_train).predict_proba(x_top_f_test)
            fpr[0], tpr[0], _ = roc_curve(y_test, probas_[:, 1])
            roc_auc[0] = auc(fpr[0], tpr[0])
            plt.figure()
            plt.plot(fpr[0], tpr[0], label='ROC curve (area = %0.2f)' % roc_auc[0])
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Plot ' + key + ' ' + algorithm  + ' Top 10 features')
            plt.legend(loc="lower right")

            plt.savefig('../features/roc_plots/' + key + '_' + algorithm + '_top10')
            plt.show()


mean_auc_file.close()