load('../data/SK_manual_kepCatRef_labelled.mat')

%%%%%%Can only have positive tags

Y = ((Y==1) + 1);

[out] = fsTtest(X, Y);
out.fList

fileID = fopen('../features/features_data/fsTtest_features_idx.txt','w');
fprintf(fileID,'Algorithm:fsTtest\n');
fprintf(fileID,'DataFile:SK_manual_kepCatRef_labelled\n');



for i = 1:size(out.fList,1),
    fprintf(fileID,'%d\t',out.fList(i) - 1);
end
fprintf('\n');
fclose(fileID);

