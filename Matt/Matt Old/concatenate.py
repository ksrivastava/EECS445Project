import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder


onehoted = np.genfromtxt('onehotencoded.tsv', skip_header=False, delimiter='\t')
reclassified = np.genfromtxt('reclassified_data.tsv', skip_header=True, delimiter='\t')
binarycontinuous = np.genfromtxt('bin_con.tsv', skip_header=True, delimiter='\t')

print str(onehoted.shape[0]) + ' by ' + str(onehoted.shape[1])
print str(reclassified.shape[0]) + ' by ' + str(reclassified.shape[1])
print str(binarycontinuous.shape[0]) + ' by ' + str(binarycontinuous.shape[1])

concatenated = np.concatenate((onehoted,reclassified,binarycontinuous), axis=1)

print 'concatenated: ' + str(concatenated.shape[0]) + ' by ' + str(concatenated.shape[1])

# imp = Imputer(missing_values=-6, strategy='median', axis = 0)
# concatenatedwithmissinghandled = imp.fit_transform(concatenated) 

# np.savetxt('final.tsv', concatenatedwithmissinghandled, delimiter='\t')
np.savetxt('final.tsv', concatenated, delimiter='\t')


# min_max_scaler = preprocessing.MinMaxScaler()

