function predictions =Stepwise_Regression_Adaboost(train_filename, test_filename)

load(train_filename);
load(train_filename);


% load('E:\Machine Learning\Final Project\kit_latest\kit\train\train.mat');
% load('E:\Machine Learning\Final Project\kit_latest\kit\test\test.mat');

X=words_train;
Y=gender_train;
[b,se,pval,inmodel,stats,nextstep,history]=stepwisefit(X,Y);

final_columns=find(inmodel==1);

data=words_train(:,final_columns);
words_train_train=data(1:4000,:);
words_train_test=data(4001:end,:);
gender_train_train=gender_train(1:4000,:);
gender_train_test=gender_train(4001:end,:);
image_features_train_train=image_features_train(1:4000,:);
image_features_train_test=image_features_train(4001:end,:);
dataspace=[words_train_train image_features_train_train];
testspace=[words_test image_features_test];

adaStump = fitensemble(dataspace,gender_train_train,'AdaBoostM1',500,'Tree',...
    'Type','Classification');
[predictions,score]=predict(adaStump,testspace);
% [r c]=size(words_train_test);
% accuracy =(sum(gender_train_test == predictions))/r;

%end 