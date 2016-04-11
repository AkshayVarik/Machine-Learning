clear
clc
load('E:\Machine Learning\Final Project\kit_latest\kit\train\train.mat');
load('E:\Machine Learning\Final Project\kit_latest\kit\test\test.mat');

result=Naive_Bayes_words(words_train,words_test,gender_train);
