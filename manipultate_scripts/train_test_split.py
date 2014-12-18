__author__ = 'yaoliu'

import numpy as np
data = np.genfromtxt('../data/preprocessed_with_one_hot.tsv', skip_header=True, delimiter='\t')
data_n = np.genfromtxt('../data/preprocessed_no_one_hot.tsv', skip_header=True, delimiter='\t')
from sklearn.cross_validation import train_test_split

from sklearn.cross_validation import KFold


n = data.shape[0]
m = data.shape[1]
x = data[:,:m-1]
y = data[:,-1]


m_n = data_n.shape[1]
x_n = data_n[:,:m_n-1]
y_n = data_n[:,-1]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)
x_n_train, x_n_test, y_n_train, y_n_test = train_test_split(x_n, y_n, test_size=0.2,random_state=42)

print m
print m_n

np.savetxt('../data/x_train_one_hot.tsv', x_train, delimiter='\t')
np.savetxt('../data/y_train_one_hot.tsv', y_train, delimiter='\t')


np.savetxt('../data/x_test_one_hot.tsv', x_test, delimiter='\t')
np.savetxt('../data/y_test_one_hot.tsv', y_test, delimiter='\t')

np.savetxt('../data/x_n_train.tsv', x_n_train, delimiter='\t')
np.savetxt('../data/y_n_train.tsv', y_n_train, delimiter='\t')

np.savetxt('../data/x_n_test.tsv', x_n_test, delimiter='\t')
np.savetxt('../data/y_n_test.tsv', y_n_test, delimiter='\t')




cv = KFold(n, n_folds=5)

for i, (train, test) in enumerate(cv):
    x_train_i =  x[train]
    x_n_train_i = x_n[train]
    y_train_i = y[train]
    y_n_train_i = y_n[train]

    x_test_i =  x[test]
    x_n_test_i = x_n[test]
    y_test_i = y[test]
    y_n_test_i = y_n[test]


    np.savetxt('../data/cv/x_train_one_hot' + str(i) + '.tsv', x_train_i, delimiter='\t')
    np.savetxt('../data/cv/y_train_one_hot' + str(i) + '.tsv', y_train_i, delimiter='\t')
    np.savetxt('../data/cv/x_test_one_hot' + str(i) + '.tsv', x_test_i, delimiter='\t')
    np.savetxt('../data/cv/y_test_one_hot' + str(i) + '.tsv', y_test_i, delimiter='\t')

    np.savetxt('../data/cv/x_n_train' + str(i) + '.tsv', x_n_train_i, delimiter='\t')
    np.savetxt('../data/cv/y_n_train' + str(i) + '.tsv', y_n_train_i, delimiter='\t')

    np.savetxt('../data/cv/x_n_test' + str(i) + '.tsv', x_n_test_i, delimiter='\t')
    np.savetxt('../data/cv/y_n_test' + str(i) + '.tsv', y_n_test_i, delimiter='\t')

