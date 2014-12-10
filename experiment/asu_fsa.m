load('../data/SK_manual_kepCatRef_labelled.mat')
Y = ((Y==1) + 1);

% 
% %%%%%%fsSBMLR Can only have positive tags
% 
% 
% [out] = fsSBMLR(X, Y);
% out.fList
% 
% fileID = fopen('../features/features_data/fsSBMLR_features_idx.txt','w');
% fprintf(fileID,'Algorithm:fsSBMLR\n');
% fprintf(fileID,'DataFile:SK_manual_kepCatRef_labelled\n');
% for i = 1:size(out.fList,2),
%     fprintf(fileID,'%d\t',out.fList(i));
% end
% fprintf('\n');
% fclose(fileID);


% % %%%%%%fsFisher 
% % 
% % 
% % [out] = fsFisher(X, Y);
% % out.fList
% % 
% % fileID = fopen('../features/features_data/fsFisher_features_idx.txt','w');
% % fprintf(fileID,'Algorithm:fsFisher\n');
% % fprintf(fileID,'DataFile:SK_manual_kepCatRef_labelled\n');
% % fprintf(fileID,'Features List\n');
% % 
% % for i = 1:size(out.fList,2),
% %     fprintf(fileID,'%d\t',out.fList(i));
% % end
% % fprintf(fileID,'\n');
% % fprintf(fileID,'Distribution\n');
% % for i = 1:size(out.W,2),
% %     fprintf(fileID,'%d\t',out.W(i));
% % end
% % fprintf('\n');
% % fclose(fileID);


% % %%%%%% fsInfoGain Can only have positive tags


[out] =  fsInfoGain(X, Y);
out.fList

fileID = fopen('../features/features_data/fsInfoGain_features_idx.txt','w');
fprintf(fileID,'Algorithm:fsInfoGain\n');
fprintf(fileID,'DataFile:SK_manual_kepCatRef_labelled\n');
fprintf(fileID,'Features List\n');

for i = 1:size(out.fList,2),
    fprintf(fileID,'%d\t',out.fList(i));
end
fprintf(fileID,'\n');
fprintf(fileID,'Gain\n');
for i = 1:size(out.W,2),
    fprintf(fileID,'%d\t',out.W(i));
end
fprintf('\n');
fclose(fileID);

