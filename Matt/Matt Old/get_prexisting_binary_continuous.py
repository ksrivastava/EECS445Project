#run  parsedata first to get proper representations for 

# !/usr/bin/env python
import csv

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

	output_file = open('bin_con.tsv', 'wb')
	data_reader = csv.DictReader(data_file, header, delimiter='\t')
	writer = csv.writer(output_file, delimiter='\t')


	data = []


	features_set = [
		'SAME_SEX_COUPLE',
		'US_RAISED',
		'ZPRURAL_CAT',
		'CORESIDENT',
		'GLBSTATUS',
		'MARRIED',
		'MET_THROUGH_AS_COWORKERS',
		'MET_THROUGH_AS_NEIGHBORS',
		'MET_THROUGH_FAMILY',
		'MET_THROUGH_FRIENDS',
		'PPHHHEAD',
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
		'PPMSACAT',
		'PPNET',
		'PARTNER_YRSED',
		'Q9',
		'RELATIONSHIP_QUALITY',
		'RESPONDENT_MOM_YRSED',
		'RESPONDENT_YRSED',
		'ZPFORBORN_CAT',
		'ZPNHBLACK_CAT',
		'ZPNHWHITE_CAT',
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
		'Q21C']
	#only append rows to data that havae some outcome 

	writer.writerow(features_set)


	for row in data_reader:
		status = checkSucc(row);        
		if status:
			#row['success'] = status
			record = []
			for f in features_set:
				if (row[f] == '-1' or row[f] == -1  or row[f] == ' ') and f != 'success':
					record.append(-6)
				else:
					record.append(row[f])
			writer.writerow(record)

	data_file.close()
	output_file.close()

	print 'pos: '+str(poscounter)
	print 'neg: '+str(negcounter)

if __name__ == '__main__':
	main()