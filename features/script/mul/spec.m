load('../../../data/matlab/one_hot_data.mat');

%%%%%%columns as classes
% pos = (y_train==1)
% neg = (y_train == -1)
% y = [pos, neg]

 W = x_train*x_train';
[ wFeat, SF ] =  fsSpectrum(W, x_train, -1);
fileID = fopen('../../features_data/mul/onehot/fsSpectrum_features_idx.txt','w');
fprintf(fileID,'Algorithm:fsSpectrum\n');
fprintf(fileID,'DataFile:one_hot_data.mat\n');

ranking =  linspace(1,138, 138);
ranking =  [ranking; wFeat']
% for i = 1:size(out.wFeat,1),
%     fprintf(fileID,'%d\t',out.fList(i) - 1);
% end

% size(out.fList)
% fprintf(fileID, '\n');
% 
% fclose(fileID);

