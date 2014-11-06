# !/usr/bin/env python
import csv

def readData(data_file, header):
	data_reader = csv.DictReader(data_file,header,delimiter='\t')
	return data_reader

def checkSucc(row):
	if (row['w4_q5'] == '1') or (row['w4_q1'] == '1'):
		return -1
		# Didn't break up

	if ((row['w4_q9']) == '1') or (row['w4_q3'] == '2') or (row['w4_q3'] == '1') or (row['W3_Q3'] == '2') or (row['W3_Q3'] == '1') or (row['W3_Q9'] == '1') or (row['W3_Q9'] == '3') or (row['W2_Q3'] == '1') or (row['W2_Q3'] == '2') or (row['W2_Q9'] == '1') or (row['W2_Q9'] == '3'):
		return 1
		# Broke up
		
	return 0

def main():
	data_file = open('../../data/30103-0001-Data.tsv', 'rU')
	header = next(data_file).strip().split('\t')
	output_file = open('../../data/kos/kos_data.txt','wb')
	data_reader = readData(data_file, header)

	output_header_file = open('../../data/kos/header.txt', 'rU')
	output_header = next(output_header_file).strip().split('\t')

	data = []
	for row in data_reader:
		status = checkSucc(row);
		if status:
			row['success'] = status
			# preprocesss
			data.append(row)

	fieldnames = sorted(list(set(k for d in data for k in d)))
	dict_writer = csv.DictWriter(output_file,fieldnames=fieldnames, delimiter='\t')
	dict_writer.writeheader()
	dict_writer.writerows(data)
	data_file.close()
	output_file.close()

if __name__ == '__main__':
    main()
