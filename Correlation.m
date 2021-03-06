clear
clc
load('E:\Machine Learning\Final Project\kit_latest\kit\train\train.mat');
load('E:\Machine Learning\Final Project\kit_latest\kit\test\test.mat');
load('E:\Machine Learning\Final Project\kit\train\comb.mat');

net_training_sum=sum(words_train);
net_training_zero=find(~net_training_sum);
net_training_nonzero=find(net_training_sum);
words_train=words_train(:,net_training_nonzero);
words_test=words_test(:,net_training_nonzero);

data = words_test;
% maxc = max(data);
% denominator = repmat(max(data),[size(data,1) 1]);
% data =data./denominator;


% Female_indices=find(gender_train);
% Male_indices=find(~gender_train);
% 
% data_male=data(Male_indices,:);
% data_female=data(Female_indices,:);
% gender_train_male=gender_train(Male_indices,:);
% gender_train_female=gender_train(Female_indices,:);

% [row col]=size(data);
% features=linspace(1,col,col);
% comb=nchoosek(features,2);

[r c]=size(comb); 
for j=1:r    
    feat1=data(:,comb(j,1));
    feat2=data(:,comb(j,2));
    Correlation_test(j)=corr(feat1,feat2);
   
%     feat1_male=data_male(:,comb(j,1));
%     feat2_male=data_male(:,comb(j,2));
%     Correlation_male(j)=corr(feat1_male,feat2_male);
%     
%     feat1_female=data_female(:,comb(j,1));
%     feat2_female=data_female(:,comb(j,2));
%     Correlation_female(j)=corr(feat1_female,feat2_female);
end



