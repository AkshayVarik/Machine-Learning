clear
clc
load('E:\Machine Learning\Final Project\kit\train\comb.mat');
load('E:\Machine Learning\Final Project\kit\train\Correlation_female.mat');
load('E:\Machine Learning\Final Project\kit\train\Correlation_male.mat');

[Correlation_female_sorted,Correlation_female_index]=sort(Correlation_female,'descend');
[Correlation_male_sorted,Correlation_male_index]=sort(Correlation_male,'descend');
thresh=0.9;
Correlation_female_above_thresh=find(Correlation_female_sorted>thresh);
Correlation_male_above_thresh=find(Correlation_male_sorted>thresh);

a=size(Correlation_female_above_thresh);
b=size(Correlation_male_above_thresh);
Correlation_female_index_above_thresh=Correlation_female_index(1:a);
Correlation_male_index_above_thresh=Correlation_male_index(1:b);

comb_female=comb(Correlation_female_index);
comb_male=comb(Correlation_male_index);