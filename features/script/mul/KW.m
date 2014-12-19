load('../../../data/matlab/one_hot_data.mat');

%%%%%%columns as classes
% pos = (y_train==1)
% neg = (y_train == -1)
% y = [pos, neg]
y_train = ((y_train==1) + 1);

[out] =  fsKruskalWallis(x_train, y_train);
fileID = fopen('../../features_data/mul/onehot/fsKruskalWallis_features_idx.txt','w');
fprintf(fileID,'Algorithm:fsKruskalWallis\n');
fprintf(fileID,'DataFile:one_hot_data.mat\n');

for i = 1:size(out.fList,1),
    fprintf(fileID,'%d\t',out.fList(i) - 1);
end

fprintf(fileID,'\n');
fprintf(fileID,'Distribution\n');
for i = 1:size(out.W,1),
    fprintf(fileID,'%d\t',out.W(i));
end

size(out.fList)
fprintf(fileID, '\n');

fclose(fileID);

