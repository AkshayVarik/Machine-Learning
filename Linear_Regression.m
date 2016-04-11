load('E:\Machine Learning\Final Project\kit\train\train.mat');
load('E:\Machine Learning\Final Project\kit\test\test.mat');
load('E:\Machine Learning\Final Project\kit\train\predictions.mat');
% load('E:\Machine Learning\Final Project\kit\train\.mat');

gender_train_test=gender_train(4001:end,:);

X=[];
for i=1:40
    X=[X predictions(i).yhat];
end

% pred_boost=[pred_abf pred_gbf pred_lbf];

Y=gender_train_test;

w=inv(X'*X)*(X'*Y);
% w_boost=inv(pred_boost'*pred_boost)*(pred_boost'*Y);
[maximum,index_max]=max(w);
[minimum,index_min]=min(w);
% [maximum_boost,index_max_boost]=max(w_boost);
% [minimum_boost,index_min_boost]=min(w_boost);

for i=1:40
    accuracy(i) =mean(gender_train_test == predictions(i).yhat)
end 


% for j=1:3
%      accuracy_boost(j) =mean(gender_train_test == pred_boost(:,j))
% end

[maximum,index_max_accuracy]=max(accuracy)
% [maximum_boost,index_max_accuracy_boost]=max(accuracy_boost)
% 
% New_predictions=pred_boost*w_boost;
% abc=New_predictions;
% New_predictions(New_predictions>0.5)=1;
% New_predictions(New_predictions<0.5)=0;
% 
% accuracy_boost_LR =mean(gender_train_test == New_predictions)
