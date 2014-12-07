import csv
import numpy

import numpy as np
from sklearn import svm
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder

def checkSucc(row):
	
	if (row['w4_q5'] == '1') or (row['w4_q1'] == '1'):
		return 1

	if ((row['w4_q9']) == '1') or (row['w4_q3'] == '2') or (row['w4_q3'] == '1') or (row['W3_Q3'] == '2') or (row['W3_Q3'] == '1') or (row['W3_Q9'] == '1') or (row['W3_Q9'] == '3') or (row['W2_Q3'] == '1') or (row['W2_Q3'] == '2') or (row['W2_Q9'] == '1') or (row['W2_Q9'] == '3'):
		return -1

	return 0

def main():
	data_file = open('30103-0001-Data.tsv', 'rU')
	header = next(data_file).strip().split('\t')
	data_reader = csv.DictReader(data_file, header, delimiter='\t')


	categorical_features = [
		'PAPGLB_FRIEND',
		'PAPRELIGION',
		'PARTNER_RACE',
		'PPMARIT',
		'PPPARTYID3',
		'PPREG9',
		'PPRENT',
		'Q12',
		'Q29',
		'Q7B',
		'RESPONDENT_RACE',
		'PAPGLB_STATUS']

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
		'Q31_1',
		'Q31_2',
		'Q31_3',
		'Q31_4',
		'Q31_5',
		'Q31_6',
		'Q31_7',
		'Q31_8',
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
		'EITHER_INTERNET_ADJUSTED' #reclassify refused -1 as missing
		]


	recode_features = [
		'PAPEVANGELICAL', #change to 0/1 from 1/2
		'Q13A',
		'Q25', #Reclassify from 1/2 to 0/1
		'Q26',  #Reclassify from 1/2 to 0/1
		'Q27',  #Reclassify from 1/2 to 0/1
		'Q28',  #Reclassify from 1/2 to 0/1
		'Q7A', #1/2 to 0/1 and -1 to empty
		'Q8A'] #1/2 to 0/1 and -1 to empty

	success = ['success']

	#categorical_features + continuous_features + binary_features + recode_features + success

	cat = []
	con = []
	bin = []
	rec = []
	stat = []

	for row in data_reader:

		status = checkSucc(row);        
		if status:

			cat_rec = []
			con_rec = []
			bin_rec = []
			recode_rec = []

			for f in categorical_features:
				if (row[f] == '-1' or row[f] == -1  or row[f] == ' ') :
					cat_rec.append(-2)
				else:
					cat_rec.append(row[f])

			for f in continuous_features:
				if (row[f] == '-1' or row[f] == -1  or row[f] == ' ') :
					con_rec.append(-2)
				else:
					con_rec.append(row[f])

			for f in binary_features:
				if (row[f] == '-1' or row[f] == -1  or row[f] == ' ') :
					bin_rec.append(-2)
				else:
					bin_rec.append(row[f])

			for f in recode_features:
				if (row[f] == '-1' or row[f] == -1  or row[f] == ' ') :
					recode_rec.append(-2)
				else:
					tmp_var = int(row[f]);
					tmp_var -= 1
					recode_rec.append(tmp_var)

			cat.append(cat_rec)
			con.append(con_rec)
			bin.append(bin_rec)
			rec.append(recode_rec)
			stat.append(status)

	num_cat = np.array(cat).astype(float)
	num_con = np.array(con).astype(float)
	num_bin = np.array(bin).astype(float)
	num_recode = np.array(rec).astype(float)
	num_status = np.array(stat).astype(float)

	print num_cat.shape
	print num_con.shape
	print num_bin.shape
	print num_recode.shape
	print num_status.shape



	#convert continuous variables based on median
	median_inputer = Imputer(missing_values=-2, strategy='median', axis = 0)
	num_con_fix = median_inputer.fit_transform(num_con)

	#convert discrete variables based on most frequent
	frequent_inputer = Imputer(missing_values=-2, strategy='most_frequent', axis = 0)
	num_cat_fix = frequent_inputer.fit_transform(num_cat) 
	num_bin_fix = frequent_inputer.fit_transform(num_bin) 
	num_recode_fix = frequent_inputer.fit_transform(num_recode) 

	one_hoter = OneHotEncoder()
	one_hoter.fit(num_cat_fix)
	num_cat_fix_one_hotted =  one_hoter.transform(num_cat_fix).toarray()
	# enc.fit(missing_filled)
	# print 'Active feature indices' 
	# print enc.active_features_
	# print 'Old to New indice mappings' 
	# print enc.feature_indices_
	# print 'Number of Encoded values' 
	# print enc.n_values_

	# one_hotted = enc.transform(missing_filled).toarray()

	# print 'dimensions after one hot ' + str(one_hotted.shape[0]) + ' by ' + str(one_hotted.shape[1]) 


	# yvalues = one_hotted[:,140];
	# print yvalues

	# valuestoscale = one_hotted[:,:139];
	# print valuestoscale

	# min_max_scaler = preprocessing.MinMaxScaler()
	# scaled = min_max_scaler.fit_transform(valuestoscale)
	# concatenated_final = np.column_stack((scaled,yvalues))

	# #fill in missing values using median

if __name__ == '__main__':
	main()