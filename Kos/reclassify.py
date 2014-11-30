import csv

def checkSucc(row):
    if (row['w4_q5'] == '1') or (row['w4_q1'] == '1'):
        return 1

    if ((row['w4_q9']) == '1') or (row['w4_q3'] == '2') or (row['w4_q3'] == '1') or (row['W3_Q3'] == '2') or (row['W3_Q3'] == '1') or (row['W3_Q9'] == '1') or (row['W3_Q9'] == '3') or (row['W2_Q3'] == '1') or (row['W2_Q3'] == '2') or (row['W2_Q9'] == '1') or (row['W2_Q9'] == '3'):
        return -1

    return 0

def main():
    data_file = open('../data/30103-0001-Data.tsv', 'rU')
    header = next(data_file).strip().split('\t')

    output_file = open('reclassified_data.tsv', 'wb')
    data_reader = csv.DictReader(data_file, header, delimiter='\t')
    writer = csv.writer(output_file, delimiter='\t')

    data = []

    features_set = ['success', 'PAPEVANGELICAL', 'Q25', 'Q26', 'Q27', 'Q28', 'Q30', 'Q7A', 'Q8A', 'S1']

    writer.writerow(features_set)

    for row in data_reader:
        status = checkSucc(row);
        if status:
            row['success'] = status
            record = []
            for f in features_set:
                if (row[f] == '-1' or row[f] == -1  or row[f] == ' ') and f != 'success':
                    record.append(-6)
                else:
                    tmp_var = int(row[f]);
                    if f != 'success':
                        if (f == 'Q30' and tmp_var == 4):
                            tmp_var = -6;
                        else:
                            tmp_var -= 1;
                    record.append(tmp_var)
            writer.writerow(record)

    data_file.close()
    output_file.close()


if __name__ == '__main__':
    main()
