import csv
import os
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

poscounter = 0;
negcounter = 0;

# Missing value = -2

def checkSucc(row):
	
	if (row['w4_q5'] == '1') or (row['w4_q1'] == '1'):
		return -1
		# -1 = they did not break up

	if ((row['w4_q9']) == '1') or (row['w4_q3'] == '2') or (row['w4_q3'] == '1') or (row['W3_Q3'] == '2') or (row['W3_Q3'] == '1') or (row['W3_Q9'] == '1') or (row['W3_Q9'] == '3') or (row['W2_Q3'] == '1') or (row['W2_Q3'] == '2') or (row['W2_Q9'] == '1') or (row['W2_Q9'] == '3'):
		return 1
		# 1 = they broke up

	return 0

def main():
	data_file = open('30103-0001-Data.tsv', 'rU')
	header = next(data_file).strip().split('\t')
	data_reader = csv.DictReader(data_file, header, delimiter='\t')

	cat_out = open('cat_out.tsv', 'wb')
	cat_writer = csv.writer(cat_out, delimiter='\t')

	con_out = open('con_out.tsv', 'wb')
	con_writer = csv.writer(con_out, delimiter='\t')

	rem_out = open('rem_out.tsv', 'wb')
	rem_writer = csv.writer(rem_out, delimiter='\t')

	suc_out = open('suc_out.tsv', 'wb')
	suc_writer = csv.writer(suc_out, delimiter='\t')

	categorical_features = [
		'PAPGLB_FRIEND', 
		'PAPRELIGION',
		'PARTNER_RACE',
		'PPMARIT',
		'PPPARTYID3',
		'PPREG9',
		'PPRENT',#refused -1
		'Q12',  #refused -1
		'Q29', #refused -1
		'Q7B', #refused -1
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
		'Q31_1', #refused is -1
		'Q31_2',#refused is -1
		'Q31_3',#refused is -1
		'Q31_4',#refused is -1
		'Q31_5',#refused is -1
		'Q31_6',#refused is -1
		'Q31_7',#refused is -1
		'Q31_8',#refused is -1
		'Q32_INTERNET',
		'SAME_SEX_COUPLE',
		'US_RAISED',
		'ZPRURAL_CAT',
		'PARENTAL_APPROVAL',
		'Q33_1',  #refused is -1
		'Q33_2',#refused is -1
		'Q33_3', #refused is -1
		'Q33_4', #refused is -1
		'Q33_5', #refused is -1
		'Q33_6', #refused is -1
		'Q33_7',#refused is -1
		'EITHER_INTERNET_ADJUSTED'  #conflicting answers -1
		]


	recode_features = [
		'PAPEVANGELICAL', #change to 0/1 from 1/2, no refused
		'Q13A', #change to 0/1 from 1/2, refused is -1
		'Q25', #change to 0/1 from 1/2, refused is -1
		'Q26',  #change to 0/1 from 1/2, refused is -1
		'Q27',  #change to 0/1 from 1/2, refused is -1
		'Q28', #change to 0/1 from 1/2, refused is -1
		'Q7A',#change to 0/1 from 1/2, refused is -1
		'Q8A'] #change to 0/1 from 1/2, refused is -1

	success = ['success']

	#writer.writerow(categorical_features + continuous_features + binary_features + recode_features + success)


	for row in data_reader:
		status = checkSucc(row);        
		if status:
			cat_record = []
			con_record = []
			rem_record = []
			suc_record = []

			for f in categorical_features:
				if (row[f] == '-1' or row[f] == -1  or row[f] == ' ') :
					cat_record.append(-1)
				else:
					cat_record.append(row[f])

			for f in continuous_features:
				if (row[f] == '-1' or row[f] == -1  or row[f] == ' ') :
					con_record.append(-1)
				else:
					con_record.append(row[f])

			for f in binary_features:

				#-1 is "probably not met online, q32 and q24 disagree", so should be reclassified as 0
				if f == 'EITHER_INTERNET_ADJUSTED' and (row[f] == '-1' or row[f] == -1):
					rem_record.append(0)
				elif (row[f] == '-1' or row[f] == -1  or row[f] == ' '):
					rem_record.append(-1)
				else:
					rem_record.append(row[f])

			for f in recode_features:
				if (row[f] == '-1' or row[f] == -1  or row[f] == ' ') :
					rem_record.append(-1)
				else:
					tmp_var = int(row[f]);
					tmp_var -= 1
					rem_record.append(tmp_var)

			suc_record.append(status)

			cat_writer.writerow(cat_record)
			con_writer.writerow(con_record)
			rem_writer.writerow(rem_record)
			suc_writer.writerow(suc_record)

	data_file.close()
	cat_out.close()
	con_out.close()
	rem_out.close()
	suc_out.close()

	# print len(overall)
	# a = numpy.array(overall)
	# print a.shape


	cat_data = np.genfromtxt('cat_out.tsv', delimiter='\t')
	con_data = np.genfromtxt('con_out.tsv', delimiter='\t')
	rem_data = np.genfromtxt('rem_out.tsv', delimiter='\t')
	suc_data = np.genfromtxt('suc_out.tsv', delimiter='\t')

	os.remove('cat_out.tsv')
	os.remove('con_out.tsv')
	os.remove('rem_out.tsv')
	os.remove('suc_out.tsv')




	# data = np.genfromtxt('all_features_manually_preprocessed.tsv',  delimiter='\t', names=True)
	print cat_data
	print cat_data.shape

	print con_data
	print con_data.shape

	print rem_data
	print rem_data.shape

	print suc_data
	print suc_data.shape

	#fill in missing values using median
	cat_imp = Imputer(missing_values=-1, strategy='most_frequent', axis = 0)
	cat_data = cat_imp.fit_transform(cat_data) 
	rem_data = cat_imp.fit_transform(rem_data) 

	con_imp = Imputer(missing_values=-1, strategy='median', axis = 0)
	con_data = con_imp.fit_transform(con_data) 

	# concatenated_before = np.concatenate((cat_data, con_data,rem_data), axis=1)
	# concatenated_before = np.column_stack((concatenated_before,suc_data))
	# np.savetxt('After_manual_preprocessing.tsv',concatenated_before, delimiter='\t')

	enc = OneHotEncoder()
	enc.fit(cat_data)
	cat_data = enc.transform(cat_data).toarray()

	print 'Active feature indices' 
	print enc.active_features_
	print 'Old to New indice mappings' 
	print enc.feature_indices_
	print 'Number of Encoded values' 
	print enc.n_values_


	min_max_scaler = preprocessing.MinMaxScaler()
	con_data = min_max_scaler.fit_transform(con_data)

	concatenated_final = np.concatenate((cat_data, con_data,rem_data), axis=1)

	print concatenated_final.shape
	concatenated_final = np.column_stack((concatenated_final,suc_data))

	np.savetxt('preprocessed.tsv', concatenated_final, delimiter = '\t')

if __name__ == '__main__':
	main()