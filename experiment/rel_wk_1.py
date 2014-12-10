__author__ = 'yaoliu'

# This script is based on background knowledge extracted from
# Earnings equality and relationship stability for same-sex and heterosexual couples, Weisshaar, Katherine
# 1. Equality of earnings reduces the likelihood of breakup for same-sex couples,
#    while it increases the likelihood of breakup for heterosexual couples  [TBD]
# 2. Couples in which the respondent has higher years of education are less likely to experience a breakup
# 3. Being married or in a domestic partnership significantly decreases the likelihood of breakup. [TBD]
# 4. The likelihood decreases with longer relationships, and with higher household incomes
# 5. TBD: Couples with children are more stable than those without children (Same-sex couples are less likely to have children)

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import linear_model

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV

# w234_combo_breakup = 1 means unsuccessful couples, i.e positive examples


# 2. Couples in which the respondent has higher years of education are less likely to experience a breakup
# 3. Being married or in a domestic partnership significantly decreases the likelihood of breakup.
# 4. The likelihood decreases with longer relationships, and with higher household incomes


data = np.genfromtxt('../data/subj_combo_breakup.tsv', skip_header=True, delimiter='\t')


header = ['w234_combo_breakup','CASEID_NEW', 'PPGENDER', 'PPAGE', 'PPINCIMP', 'HOW_LONG_AGO_FIRST_ROMANTIC',
                    'SAME_SEX_COUPLE', 'Q23', 'Q19', 'PPEDUC', 'Q18A_3', 'MET_THROUGH_FRIENDS',
                    'MET_THROUGH_FAMILY', 'MET_THROUGH_AS_NEIGHBORS', 'MET_THROUGH_AS_COWORKERS', 'Q24_CHURCH',
                    'Q24_SCHOOL', 'Q24_COLLEGE', 'Q31_1', 'Q31_2', 'Q31_3', 'Q31_4', 'Q31_5', 'Q31_6', 'Q31_7', 'Q31_8',
                    'Q31_9', 'Q31_OTHER_TEXT_ENTERED']

extract_set_1 = ['w234_combo_breakup','PPAGE']

extract_set_2 = ['w234_combo_breakup','PPEDUC']

extract_set_3 = ['w234_combo_breakup','HOW_LONG_AGO_FIRST_ROMANTIC']

extract_set_4 = ['w234_combo_breakup','PPINCIMP']

extract_set_1_2 = ['w234_combo_breakup','PPAGE', 'PPEDUC']

extract_set_1_2_3 = ['w234_combo_breakup','PPAGE', 'PPEDUC', 'HOW_LONG_AGO_FIRST_ROMANTIC']

extract_set_1_2_3_4 = ['w234_combo_breakup','PPAGE', 'PPEDUC', 'HOW_LONG_AGO_FIRST_ROMANTIC','PPINCIMP' ]

extract_set_5 = ['w234_combo_breakup', 'MET_THROUGH_FRIENDS', 'MET_THROUGH_FAMILY', 'MET_THROUGH_AS_NEIGHBORS','MET_THROUGH_AS_COWORKERS', 'Q24_CHURCH', 'Q24_SCHOOL', 'Q24_COLLEGE']

extract_set_6 = ['w234_combo_breakup', 'PPAGE', 'PPEDUC', 'HOW_LONG_AGO_FIRST_ROMANTIC','PPINCIMP','MET_THROUGH_FRIENDS', 'MET_THROUGH_FAMILY', 'MET_THROUGH_AS_NEIGHBORS','MET_THROUGH_AS_COWORKERS', 'Q24_CHURCH', 'Q24_SCHOOL', 'Q24_COLLEGE' ]

extract_set_7 = ['w234_combo_breakup', 'Q18A_3']

extract_set_comb = [extract_set_1, extract_set_2, extract_set_3, extract_set_4, extract_set_1_2, extract_set_1_2_3, extract_set_1_2_3_4, extract_set_5, extract_set_6, extract_set_7]


testing_scores_set = []
training_scores_set = []
ratio = []

nb_testing_scores_set = []
nb_training_scores_set = []

svm_testing_scores_set = []
svm_training_scores_set = []

knn_testing_scores_set = []
knn_training_scores_set = []


for s in extract_set_comb:
    index = []
    for f in s:
        index.append(header.index(f))
    extract_data = data[:,index]
    #remove missing values
    extract_data = extract_data[~np.isnan(extract_data).any(1)]
    
    y = extract_data[:,0]
    x = extract_data[:,1:]
    ratio.append(1 - sum(y)/len(y))
    y = 2 * y - 1
    i = 0
    testing_scores = []
    training_scores = []
    while i < 5:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=np.random.RandomState())
        logreg = linear_model.LogisticRegression().fit(x_train, y_train)
        scores = cross_validation.cross_val_score(logreg, x_train,  y_train, cv=5)
        training_scores.append(scores.mean())
        testing_scores.append(logreg.score(x_test, y_test))
        i += 1
    testing_scores_set.append(sum(testing_scores)/5)
    training_scores_set.append(sum(training_scores)/5)


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=np.random.RandomState())

    #############Adapted from Kos's script############################
    gnb = GaussianNB()
    parameters = {'kernel':('linear', 'rbf'), 'C':[0.001, 1, 100], 'gamma': [0.001, 0.0001]}
    svr = svm.SVC()
    clf = GridSearchCV(estimator=svr, param_grid=parameters, cv=2)
    parameters = {'n_neighbors':[4, 12, 20], 'algorithm':('auto', 'ball_tree', 'kd_tree', 'brute')}
    kvr = KNeighborsClassifier()
    knn = GridSearchCV(estimator=kvr, param_grid=parameters)

    gnb.fit(x_train, y_train)
    clf.fit(x_train, y_train)
    knn.fit(x_train, y_train)

    nb_training_scores_set.append(gnb.score(x_train, y_train))
    svm_training_scores_set.append(clf.score(x_train, y_train))
    knn_training_scores_set.append(knn.score(x_train, y_train))

    nb_testing_scores_set.append(gnb.score(x_test, y_test))
    svm_testing_scores_set.append(clf.score(x_test, y_test))
    knn_testing_scores_set.append(knn.score(x_test, y_test))


print "Ratio of negative examples"
print ratio
print "Logistic regression"
print training_scores_set
print testing_scores_set
print "Naive Bayes"
print nb_training_scores_set
print nb_testing_scores_set
print "SVM"
print svm_training_scores_set
print svm_testing_scores_set
print "KNN"
print knn_training_scores_set
print knn_testing_scores_set








