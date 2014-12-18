load('../../../data/matlab/data_without_one_hot.mat');
%%%%%%Can only have positive tags
ny_train = ((ny_train==1) + 1);
[out] = fsTtest(nx_train, ny_train);
fileID = fopen('../../features_data/univariate/no_one_hot/nfsTtest_features_idx.txt','w');
fprintf(fileID,'Algorithm:fsTtest\n');
fprintf(fileID,'DataFile:data_without_one_hot.mat\n');
for i = 1:size(out.fList,1),
    fprintf(fileID,'%d\t',out.fList(i) - 1);
end
size(out.fList)
fprintf('\n');
fclose(fileID);
