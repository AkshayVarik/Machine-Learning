clear
clc
load('E:\Machine Learning\Final Project\kit_latest\kit\train\train.mat');
load('E:\Machine Learning\Final Project\kit_latest\kit\test\test.mat');

net_training_sum=sum(words_train);
net_training_zero=find(~net_training_sum);
net_training_nonzero=find(net_training_sum);
words_train=words_train(:,net_training_nonzero);
words_test=words_test(:,net_training_nonzero);

% words_train_train=words_train(1:4000,:);
% words_train_test=words_train(4001:4998,:);
% genders_train_train=gender_train(1:4000,:);
% genders_train_test=gender_train(4001:4998,:);


% W_MLE=inv(words_train_train'*words_train_train)*(words_train_train'*genders_train_train);
% ERROR=[norm(genders_train_train-words_train_train*W_MLE)]^2;
% 
% % Common=@(w)[norm(genders_train_train-words_train_train*w)]^2+[norm(w)]^2;
% % [w,fval]=fminsearch(Common,[W_MLE]);
% % ERROR1=[norm(genders_train_train-words_train_train*w)]^2;
% 
% Common=@(w)[norm(genders_train_train-words_trai n_train*w)]^2+[norm(w,1)]^1;
% [w,fval]=fminsearch(Common,[W_MLE]);
% ERROR2=[norm(genders_train_train-words_train_train*w)]^2;


data = words_train;
labels = gender_train;
ratio = 0.75;

for j=1:20
    j
    idx = randperm(size(data,1));
    numInst = size(data,1);
    numTrain = floor(ratio*numInst); numVal = numInst - numTrain;
    words_train_train = data(idx(1:numTrain),:); 
    words_train_test = data(idx(numTrain+1:end),:);    
    genders_train_train = labels(idx(1:numTrain)); 
    genders_train_test= labels(idx(numTrain+1:end));
    

% cvpart = cvpartition(genders_train,'holdout',0.3);
% words_train_train = words_train(training(cvpart),:);
% gender_train_train = genders_train(training(cvpart),:);
% words_train_test = words_train(test(cvpart),:);
% gender_train_test = genders_train(test(cvpart),:);

adaStump = fitensemble(words_train_train,genders_train_train,'AdaBoostM1',500,'Tree',...
    'Type','Classification');
label(:,j)=predict(adaStump,words_train_test);

% figure;
% plot(resubLoss(adaStump,'Mode','Cumulative'));
% xlabel('Number of trees');
% ylabel('Test classification error');

[r c]=size(words_train_test);
accuracy(:,j) =(sum(genders_train_test == label(:,j)))/r

end
% c = cvpartition(genders_train_train,'k',10);
% opts = statset('display','iter');
% fun = @(words_train_train,genders_train_train,words_train_test,genders_train_test)...
%       (sum(~strcmp(genders_train_test,classify(words_train_test,words_train_train,genders_train_test,'quadratic'))));
% 
% [fs,history] = sequentialfs(fun,words_train,gender_train,'cv',c,'options',opts)

% [ranked,weights]=relieff(words_train_train,genders_train_train,10);
% bar(weights(ranked));
% xlabel('Predictor rank');
% ylabel('Predictor importance weight');
