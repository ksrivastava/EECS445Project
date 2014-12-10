__author__ = 'yaoliu'
# This script is to test the drawing of roc curve
print(__doc__)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from scipy import interp

data = np.genfromtxt('../data/subj_combo_breakup.tsv', skip_header=True, delimiter='\t')
data = data[~np.isnan(data).any(1)]

header = ['w234_combo_breakup','CASEID_NEW', 'PPGENDER', 'PPAGE', 'PPINCIMP', 'HOW_LONG_AGO_FIRST_ROMANTIC',
                    'SAME_SEX_COUPLE', 'Q23', 'Q19', 'PPEDUC', 'Q18A_3', 'MET_THROUGH_FRIENDS',
                    'MET_THROUGH_FAMILY', 'MET_THROUGH_AS_NEIGHBORS', 'MET_THROUGH_AS_COWORKERS', 'Q24_CHURCH',
                    'Q24_SCHOOL', 'Q24_COLLEGE', 'Q31_1', 'Q31_2', 'Q31_3', 'Q31_4', 'Q31_5', 'Q31_6', 'Q31_7', 'Q31_8',
                    'Q31_9', 'Q31_OTHER_TEXT_ENTERED']

y = data[:,0]
x = data[:,1:]

ratio = (1 - sum(y)/len(y))

y = 2 * y - 1

testing_scores = 0
training_scores = 0

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=np.random.RandomState())

############# No Cross Validation ############################


#########For Single Model

#
# # Compute ROC curve and ROC area for a specific model


logregClassifier = linear_model.LogisticRegression()

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
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

#
# # Compute ROC curve and ROC area for a multiple models

nb = GaussianNB()
svr = svm.SVC(probability=True)
kvr = KNeighborsClassifier()

scores = dict()


fpr_mul = dict()
tpr_mul = dict()
roc_auc_mul = dict()
probas = dict()

probas[0]  =  nb.fit(x_train, y_train).predict_proba(x_test)
probas[1]  =  svr.fit(x_train, y_train).predict_proba(x_test)
probas[2]  =  kvr.fit(x_train, y_train).predict_proba(x_test)

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)


for i in range(3):
    fpr_mul[i] ,tpr_mul[i], _ = roc_curve(y_test,  (probas[i])[:, 1])
    mean_tpr += interp(mean_fpr, fpr_mul[i] ,tpr_mul[i])
    mean_tpr[0] = 0.0
    roc_auc_mul[i] = auc(fpr_mul[i] ,tpr_mul[i])

mean_tpr /= 3
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)


# Plot ROC curve
plt.figure()

plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

for i in range(3):
    plt.plot(fpr_mul[i], tpr_mul[i], label='ROC curve of classifier {0} (area = {1:0.2f})'
                                   ''.format(i, roc_auc_mul[i]))


plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(' Receiver operating characteristic to multi-models')
plt.legend(loc="lower right")
plt.show()


############# Cross Validation ############################

random_state = np.random.RandomState(0)
cv = StratifiedKFold(y, n_folds=5)
classifier = GaussianNB()



mean_tpr_cv = 0.0
mean_fpr_cv = np.linspace(0, 1, 100)

for i, (train, test) in enumerate(cv):
    probas_ = classifier.fit(x[train], y[train]).predict_proba(x[test])
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
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


