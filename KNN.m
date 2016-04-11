% function predictions =KNN(train_filename, test_filename)
% 
% load(train_filename);
% load(test_filename);


load('E:\Machine Learning\Final Project\kit_latest\kit\train\train.mat');
load('E:\Machine Learning\Final Project\kit_latest\kit\test\test.mat');


words_train_train=words_train(1:4000,:);
words_train_test=words_train(4001:end,:);
gender_train_train=gender_train(1:4000,:);
gender_train_test=gender_train(4001:end,:);
image_features_train_train=image_features_train(1:4000,:);
image_features_train_test=image_features_train(4001:end,:);
data=[words_train_train image_features_train_train];
testspace=[words_test image_features_test];

mdl=fitcknn(data,gender_train_train);
[predictions,score]=predict(mdl,testspace);
[r c]=size(words_train_test);
accuracy =(sum(gender_train_test == predictions))/r;

% end