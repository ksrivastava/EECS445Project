__author__ = 'yaoliu'

import csv
feature_dir = '../features/features_data/'


infile = open(feature_dir + 'concat_feature_list.txt', 'r')
outfile = open(feature_dir + 'cont_cat_fsa_list.txt','w')

bl = next(infile).strip().split('\t')

i = 0
record = []
for item in bl:
    if int(item):
        record.append(i)
    i = i + 1

csv_writer = csv.writer(outfile, delimiter='\t')
csv_writer.writerow(record)
outfile.close()
infile.close()


