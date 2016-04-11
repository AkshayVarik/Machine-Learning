load('E:\Machine Learning\Final Project\kit_latest\kit\train\train.mat');

% rng(8000,'twister');
% 
% [row col]=size(words_train);
% 
% holdoutCVP=cvpartition(row,'holdout',998);
% dataTrain = obs(holdoutCVP.training,:);
% grpTrain = grp(holdoutCVP.training);

insight=sum(words_train(:,:)==0);
index=[];
for i=1:5000
    if insight(i)<3000
        index=[index i];
    end 
end 
words_train_reduced=words_train(:,index);
% [row col]=size(words_train_reduced);

words_train_reduced_train=words_train_reduced(1:4000,:);
words_train_reduced_test=words_train_reduced(4001:4998,:);
gender_train_train=gender_train(1:4000,:);
gender_train_test=gender_train(4001:4998,:);

[r c]=size(words_train_reduced_test);

mdl=fitcnb(words_train_reduced_train,gender_train_train,'Distribution','mn');
predictlabel=predict(mdl,words_train_reduced_test);
accuracy =(sum(gender_train_test == predictlabel))/r