import csv
import numpy

poscounter = 0;
negcounter = 0;

def checkSucc(row):
	
	if (row['w4_q5'] == '1') or (row['w4_q1'] == '1'):
		global poscounter
		# if poscounter == 450:
		# 	return 0
		poscounter+=1
		return 1

	if ((row['w4_q9']) == '1') or (row['w4_q3'] == '2') or (row['w4_q3'] == '1') or (row['W3_Q3'] == '2') or (row['W3_Q3'] == '1') or (row['W3_Q9'] == '1') or (row['W3_Q9'] == '3') or (row['W2_Q3'] == '1') or (row['W2_Q3'] == '2') or (row['W2_Q9'] == '1') or (row['W2_Q9'] == '3'):
		global negcounter
		negcounter+= 1
		return -1

	return 0

def main():
	data_file = open('30103-0001-Data.tsv', 'rU')
	header = next(data_file).strip().split('\t')

	output_file = open('all_features_manually_preprocessed.tsv', 'wb')
	data_reader = csv.DictReader(data_file, header, delimiter='\t')
	writer = csv.writer(output_file, delimiter='\t')


	data = []


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

	writer.writerow(categorical_features + continuous_features + binary_features + recode_features + success)

	overall = []

	for row in data_reader:
		status = checkSucc(row);        
		if status:
			record = []

			for f in categorical_features:
				if (row[f] == '-1' or row[f] == -1  or row[f] == ' ') :
					record.append(-2)
				else:
					record.append(row[f])

			for f in continuous_features:
				if (row[f] == '-1' or row[f] == -1  or row[f] == ' ') :
					record.append(-2)
				else:
					record.append(row[f])

			for f in binary_features:
				if (row[f] == '-1' or row[f] == -1  or row[f] == ' ') :
					record.append(-2)
				else:
					record.append(row[f])

			for f in recode_features:
				if (row[f] == '-1' or row[f] == -1  or row[f] == ' ') :
					record.append(-2)
				else:
					tmp_var = int(row[f]);
					tmp_var -= 1
					record.append(tmp_var)

			record.append(status)
			print len(record)

			overall.append(record)
			writer.writerow(record)

	data_file.close()
	output_file.close()

	print len(overall)
	a = numpy.array(overall)
	print a.shape
	print 'pos: '+str(poscounter)
	print 'neg: '+str(negcounter)

if __name__ == '__main__':
	main()