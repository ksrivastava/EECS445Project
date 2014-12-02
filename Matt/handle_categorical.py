import csv
import numpy as np
from sklearn import preprocessing
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

	output_file = open('cat_data_to_be_1hot.tsv', 'wb')
	writer = csv.writer(output_file, delimiter='\t')

	data = []
	features_set = ['PAPGLB_FRIEND','PAPRELIGION','PARTNER_RACE', 'PPMARIT','PPPARTYID3','PPREG9','PPRENT','Q12','Q29','Q7B','RESPONDENT_RACE']

	writer.writerow(features_set)

	for row in data_reader:
		status = checkSucc(row);        
		if status:
			record = []
			for f in features_set:
				if (row[f] == '-1' or row[f] == -1  or row[f] == ' '):
					record.append(-6)
				else:
					record.append(row[f])
			writer.writerow(record)

	data_file.close()
	output_file.close()

	data = np.genfromtxt('cat_data_to_be_1hot.tsv', skip_header=True, delimiter='\t')

	imp = Imputer(missing_values=-6, strategy='median', axis = 0)
	complete = imp.fit_transform(data) 

	enc = OneHotEncoder()
	features = enc.fit_transform(complete).toarray()

	np.savetxt('onehotencoded.tsv', features, delimiter='\t')

	m = features.shape[1]
	n = features.shape[0]
	print 'dimensions ' + str(m) + ' by ' + str(n) 


if __name__ == '__main__':
	main()
