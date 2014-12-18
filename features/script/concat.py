__author__ = 'yaoliu'




import fnmatch
import os
import numpy as np
import linecache
import csv

features_list = [0]*143

dir = '../features_data/'
for file in os.listdir(dir):
    if fnmatch.fnmatch(file, '*_idx.txt'):
        print file
    #TODO: encode features lines number into file names


i  = 0
for file in os.listdir(dir):
    if fnmatch.fnmatch(file, '*_idx.txt'):
        idx = linecache.getline(dir + file,3).strip().split('\t')
        idx_nu =  map(int, idx)
        top_10 = idx_nu[:10]

        for f in top_10:
            features_list[f] = 1
print sum(features_list)

fout = open(dir + 'concat_feature_list.txt', 'wb')
csv_writer = csv.writer(fout, delimiter='\t')
csv_writer.writerow(features_list)
fout.close()

