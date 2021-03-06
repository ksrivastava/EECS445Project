load('../../../data/matlab/one_hot_data.mat');

%%%%%%Can only have positive tags
y_train = ((y_train==1) + 1);
[out] = fsTtest(x_train, y_train);
fileID = fopen('../../features_data/univariate/onehot/fsTtest_features_idx.txt','w');
fprintf(fileID,'Algorithm:fsTtest\n');
fprintf(fileID,'DataFile:one_hot_data.mat\n');
for i = 1:size(out.fList,1),
    fprintf(fileID,'%d\t',out.fList(i) - 1);
end
size(out.fList)
fprintf(fileID, '\n');

fprintf(fileID,'Distribution\n');
for i = 1:size(out.fList,1),
    fprintf(fileID,'%d\t',out.W(i));
end


