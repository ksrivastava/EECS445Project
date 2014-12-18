__author__ = 'yaoliu'


import fnmatch
import os
import numpy as np
import linecache
import csv

features_list = [0]*144

dir = '../features_data/'
for file in os.listdir(dir):
    if fnmatch.fnmatch(file, '*_idx.txt'):
        print file
    #TODO: encode features lines number into file names

weight = [0.9, 0.9, 0.9, 0.82, 0.82]
weight_20 = [0.9, 0.9, 0.9, 0.82, 0.82]

i  = 0
for file in os.listdir(dir):
    if fnmatch.fnmatch(file, '*_idx.txt'):
        idx = linecache.getline(dir + file,3).strip().split('\t')
        idx_nu =  map(int, idx)
        ranking = 143 * weight[i]
        for f in idx_nu:
            features_list[f] = features_list[f] + ranking
            ranking = ranking - weight[i]
        i += 1

arr = np.array(features_list)

fout = open(dir + 'ranking_features_list.txt', 'wb')
csv_writer = csv.writer(fout, delimiter='\t')

csv_writer.writerow(np.argsort(arr)[::-1][:144])
fout.close()

