xtrain_f = '../data/x_train_one_hot.tsv';
ytrain_f = '../data/y_train_one_hot.tsv';
xtest_f = '../data/x_test_one_hot.tsv';
ytest_f = '../data/y_test_one_hot.tsv';

delimiterIn = '\t';
x_train = importdata(xtrain_f,delimiterIn);
y_train = importdata(ytrain_f,delimiterIn);
x_test = importdata(xtest_f,delimiterIn);
y_test = importdata(ytest_f,delimiterIn);

save('../data/matlab/one_hot_data.mat','x_train','y_train','x_test','y_test');

%%%%%%%%%%%%%%%%%%% For Non One Hot Encoding
nxtrain_f = '../data/x_n_train.tsv';
nytrain_f = '../data/y_n_train.tsv';
nxtest_f = '../data/x_n_test.tsv';
nytest_f = '../data/y_n_test.tsv';


nx_train = importdata(nxtrain_f,delimiterIn);
size(nx_train)
ny_train = importdata(nytrain_f,delimiterIn);
nx_test = importdata(nxtest_f,delimiterIn);
ny_test = importdata(nytest_f,delimiterIn);

save('../data/matlab/data_without_one_hot.mat','nx_train','ny_train','nx_test','ny_test');