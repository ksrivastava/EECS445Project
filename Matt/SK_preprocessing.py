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
print 'original dimensions ' + str(data.shape[0]) + ' by ' + str(data.shape[1]) 


#fill in missing values using median
imp = Imputer(missing_values=-1, strategy='median', axis = 0)
missing_filled = imp.fit_transform(data) 

#convert categorical features into multiple binary sets
cat_feature_index = list(xrange(12))
print cat_feature_index

np.savetxt('missing_filled.tsv', missing_filled, delimiter = '\t')

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


np.savetxt('SK_preprocessed.tsv', one_hotted, delimiter = '\t')


min_max_scaler = preprocessing.MinMaxScaler()
scaled = min_max_scaler.fit_transform(one_hotted)

np.savetxt('SK_scaled.tsv', scaled, delimiter = '\t')
