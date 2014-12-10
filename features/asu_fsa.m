load('../data/SK_manual_kepCatRef_labelled.mat')

%%%%%%Can only have positive tags

Y = ((Y==1) + 1);

[out] = fsSBMLR(X, Y);
out.fList

fileID = fopen('../features/fsSBMLR_features_idx.txt','w');
fprintf(fileID,'Algorithm:fsSBMLR\n');
fprintf(fileID,'DataFile:SK_manual_kepCatRef_labelled\n');
for i = 1:size(out.fList,1),
    fprintf(fileID,'%d\t',out.fList(i));
end
fprintf('\n');
fclose(fileID);

