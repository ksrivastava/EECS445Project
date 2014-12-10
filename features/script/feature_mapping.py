####################
#	EECS445 Project
#	Filename: feature_mapping.py
#	Author: Y.L.
#	Created: Dec 10
####################

import linecache
import csv
import os
# REQUIRE:
	# header_line and idx_line specifies the line number, 1 based, tab seperated
	# header_file and idx_file specifies the input file name
	# write file is output file name
# EFFECT: 
	# print features and write into file

def map_features(header_file, header_line, idx_file, idx_line, write_file):
	header_line = int(header_line)
	idx_line = int(idx_line)


	header = linecache.getline(header_file,header_line).strip().split('\t')
	print header
	print(len(header))
	idx = linecache.getline(idx_file,idx_line).strip().split('\t')
	out = open(write_file, 'wb')
	writer = csv.writer(out, delimiter='\t')
	print idx
	ifile = open(idx_file, 'r')
	lines = ifile.readlines()
	for line in lines:
		line = line.strip().split('\t')
		writer.writerow(line)
	features = []

	for f in idx:
         f = int(f)
         features.append(header[f])

	writer.writerow(features)
	out.close()
	ifile.close()
	print features
	
