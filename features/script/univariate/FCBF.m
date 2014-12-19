load('../../../data/matlab/one_hot_data.mat');

%%%%%%Can only have positive tags
y_train = ((y_train==1) + 1);
[out] =  fsFCBF(x_train, y_train);
fileID = fopen('../../features_data/mul/onehot/fsFCBF_features_idx.txt','w');
fprintf(fileID,'Algorithm:fsFCBF\n');
fprintf(fileID,'DataFile:one_hot_data.mat\n');

for i = 1:size(out.fList,2),
    fprintf(fileID,'%d\t',out.fList(i) - 1);
end

size(out.fList)
fprintf(fileID, '\n');



fclose(fileID);


