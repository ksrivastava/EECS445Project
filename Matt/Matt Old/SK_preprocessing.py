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

data = np.genfromtxt('all_features_manually_preprocessed.tsv', skip_header=True, delimiter='\t')

# data = np.genfromtxt('all_features_manually_preprocessed.tsv',  delimiter='\t', names=True)
print data
print data.shape

#fill in missing values using median
imp = Imputer(missing_values=-2, strategy='most_frequent', axis = 0)
missing_filled = imp.fit_transform(data) 

#convert categorical features into multiple binary sets
cat_feature_index = list(xrange(12))
print cat_feature_index


print missing_filled.shape

enc = OneHotEncoder(categorical_features=cat_feature_index)

enc.fit(missing_filled)
print 'Active feature indices' 
print enc.active_features_
print 'Old to New indice mappings' 
print enc.feature_indices_
print 'Number of Encoded values' 
print enc.n_values_

one_hotted = enc.transform(missing_filled).toarray()

print 'dimensions after one hot ' + str(one_hotted.shape[0]) + ' by ' + str(one_hotted.shape[1]) 


m = data.shape[1] - 1;
n = data.shape[0];

yvalues = one_hotted[:,140];
print yvalues

valuestoscale = one_hotted[:,:139];
print valuestoscale

min_max_scaler = preprocessing.MinMaxScaler()
scaled = min_max_scaler.fit_transform(valuestoscale)
concatenated_final = np.column_stack((scaled,yvalues))

np.savetxt('SK_final.tsv', concatenated_final, delimiter = '\t')
