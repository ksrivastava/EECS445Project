# __author__ = 'yaoliu'


## Adapt from Matt ######
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.preprocessing import Imputer


data = np.genfromtxt('../data/all_features_manually_preprocessed_labeled_keepCatRef.tsv', skip_header=True, delimiter='\t')
print 'original dimensions ' + str(data.shape[0]) + ' by ' + str(data.shape[1])


#fill in missing values using median
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis = 0)
missing_filled = imp.fit_transform(data)

#convert categorical features into multiple binary sets
cat_feature_index = list(xrange(12))
print cat_feature_index

categorical_features = [
        'PAPGLB_FRIEND',
        'PAPRELIGION',
        'PARTNER_RACE',
        'PPMARIT',
        'PPPARTYID3',
        'PPREG9',
        'PPRENT',
        'Q12', # Have refused as -1
        'Q29', # Have refused as -1
        'Q7B', # Have refused as -1
        'RESPONDENT_RACE',
        'PAPGLB_STATUS'] # Have refused as 3

continuous_features = [
        'AGE_DIFFERENCE',
        'CHILDREN_IN_HH',
        'DISTANCEMOVED_10MI',
        'GENDER_ATTRACTION',
        'HHINC',
        'HOW_LONG_AGO_FIRST_COHAB',
        'HOW_LONG_AGO_FIRST_MET',
        'HOW_LONG_AGO_FIRST_ROMANTIC',
        'HOW_LONG_RELATIONSHIP',
        'PARTNER_MOM_YRSED',
        'PARTNER_YRSED',
        'PPAGECAT',
        'PPHOUSE',
        'PPHOUSEHOLDSIZE',
        'Q21A',
        'Q21B',
        'Q21C',
        'Q9',
        'RELATIONSHIP_QUALITY',
        'RESPONDENT_MOM_YRSED',
        'RESPONDENT_YRSED',
        'ZPFORBORN_CAT',
        'ZPNHBLACK_CAT',
        'ZPNHWHITE_CAT']

binary_features = [
        'CORESIDENT',
        'GLBSTATUS',
        'MARRIED',
        'MET_THROUGH_AS_COWORKERS',
        'MET_THROUGH_AS_NEIGHBORS',
        'MET_THROUGH_FAMILY',
        'MET_THROUGH_FRIENDS',
        'PPHHHEAD',
        'PPMSACAT',
        'PPNET',
        'Q24_MET_ONLINE',
        'Q31_1', #refused -1
        'Q31_2', #refused -1
        'Q31_3', #refused -1
        'Q31_4', #refused -1
        'Q31_5', #refused -1
        'Q31_6', #refused -1
        'Q31_7', #refused -1
        'Q31_8', #refused -1
        'Q32_INTERNET',
        'SAME_SEX_COUPLE',
        'US_RAISED',
        'ZPRURAL_CAT',
        'PARENTAL_APPROVAL',
        'Q33_1',  #reclassify refused -1 as missing
        'Q33_2', #reclassify refused -1 as missing
        'Q33_3', #reclassify refused -1 as missing
        'Q33_4', #reclassify refused -1 as missing
        'Q33_5', #reclassify refused -1 as missing
        'Q33_6', #reclassify refused -1 as missing
        'Q33_7',#reclassify refused -1 as missing
        'EITHER_INTERNET_ADJUSTED' #reclassify -1 as 0 (probably not met online)
        ]


recode_features = [
        'PAPEVANGELICAL', #change to 0/1 from 1/2
        'Q13A', #refused -1
        'Q25', #Reclassify from 1/2 to 0/1, #refused -1
        'Q26',  #Reclassify from 1/2 to 0/1
        'Q27',  #Reclassify from 1/2 to 0/1 #refused -1
        'Q28',  #Reclassify from 1/2 to 0/1 #refused -1
        'Q7A', #1/2 to 0/1 and -1 to empty #refused -1
        'Q8A'] #1/2 to 0/1 and -1 to empty #refused -1


success = ['SUCCESS']

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




new_cat = []
sum = np.sum(enc.n_values_)
k = 0
cat_feature_index = list(xrange(12))

for i in range(0,len(cat_feature_index)):
    k = -1
    for j in range(enc.feature_indices_[i],enc.feature_indices_[i + 1]):
        k  += 1
        if j not in enc.active_features_:
            continue
        new_cat.append(categorical_features[i] + '_' + str(k))


header_list = new_cat + continuous_features + binary_features + recode_features + success
header_names = '\t'.join(header_list)

continuous_l = len(new_cat)
continuous_u = continuous_l + len(continuous_features)



valuestoscale = one_hotted[:,continuous_l:continuous_u]
print valuestoscale

pre = one_hotted[:,0:continuous_l]
po = one_hotted[:,continuous_u:]

# print yvalues

nm_scaler = preprocessing.StandardScaler()

scaled = nm_scaler.fit_transform(valuestoscale)
concatenated_final = np.column_stack((pre, scaled, po))
print(concatenated_final.shape[1])
print(one_hotted.shape[1])


np.savetxt('../data/SK_manual_kepCatRef_labelled.tsv', concatenated_final, delimiter = '\t', header = header_names)


# 0 0 3
# 1 1 0
# 0 2 1
# 1 0 2
#
# from sklearn.preprocessing import OneHotEncoder
# enc = OneHotEncoder()
# enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
# OneHotEncoder(categorical_features='all', dtype='float', n_values='auto', sparse=True)
# print enc.n_values_
# # array([2, 3, 4])
# print enc.feature_indices_
# # array([0, 2, 5, 9])
# print enc.transform([[0, 1, 1]]).toarray()
# # array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.]])
# print enc.active_features_
