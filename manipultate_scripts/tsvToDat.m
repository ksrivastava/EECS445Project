filename = '../data/SK_manual_kepCatRef_labelled.tsv';
delimiterIn = '\t';
headerlinesIn = 1;
A = importdata(filename,delimiterIn,headerlinesIn);
[n, m] = size(A.data);
X = A.data(:,1:m-1);
Y = A.data(:,m);

save('../data/SK_manual_kepCatRef_labelled.mat','X','Y')