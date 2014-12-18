__author__ = 'yaoliu'

# This script is to select features based on the subjective features selection method
# based on background knowledge we extract from relevant social science research  work

# !/usr/bin/env python
import csv


def read_data(data_file, header):
    data_reader = csv.DictReader(data_file, header, delimiter='\t')
    return data_reader


def main():
    data_file = open('../data/classify_suc_or_not_missing.tsv', 'rU')
    header = next(data_file).strip().split('\t')
    output_file = open('../data/subj.tsv', 'wb')
    data_reader = read_data(data_file, header)
    writer = csv.writer(output_file, delimiter='\t')
# 'CASEID_NEW',
    features_set = ['PPGENDER', 'PPAGE', 'PPINCIMP', 'HOW_LONG_AGO_FIRST_ROMANTIC',
                    'SAME_SEX_COUPLE', 'Q23', 'Q19', 'PPEDUC', 'Q18A_3', 'MET_THROUGH_FRIENDS',
                    'MET_THROUGH_FAMILY', 'MET_THROUGH_AS_NEIGHBORS', 'MET_THROUGH_AS_COWORKERS', 'Q24_CHURCH',
                    'Q24_SCHOOL', 'Q24_COLLEGE', 'Q31_1', 'Q31_2', 'Q31_3', 'Q31_4', 'Q31_5', 'Q31_6', 'Q31_7', 'Q31_8',
                    'Q31_9', 'Q31_OTHER_TEXT_ENTERED','SUCCESS']

    header =  features_set

    writer.writerow(header)
    for row in data_reader:
        record = []
        for f in features_set:
            record.append(row[f])
        writer.writerow(record)

    data_file.close()
    output_file.close()


if __name__ == '__main__':
    main()