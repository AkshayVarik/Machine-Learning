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


holdoutCVP=cvpartition(gender_train,'holdout',755);
words_train_train=words_train(holdoutCVP.training,:);
gender_train_train=gender_train(holdoutCVP.training);

words_train_train_G1=words_train_train(grp2idx(gender_train_train)==1,:);
words_train_train_G2=words_train_train(grp2idx(gender_train_train)==2,:);
[h,p,ci,stat] = ttest2(words_train_train_G1,words_train_train_G2,'Vartype','unequal');

ecdf(p);
xlabel('P value');
ylabel('CDF value');


[~,featureIdxSortbyP] = sort(p,2); % sort the features
testMCE = zeros(1,14);
resubMCE = zeros(1,14);
nfs = 5:5:1000;
classf = @(xtrain,ytrain,xtest,ytest) ...
             sum(~strcmp(ytest,classify(xtest,xtrain,ytrain,'quadratic')));
resubCVP = cvpartition(length(gender_train),'resubstitution')
% for i = 1:200
%    fs = featureIdxSortbyP(1:nfs(i));
%    testMCE(i) = crossval(classf,words_train(:,fs),gender_train,'partition',holdoutCVP)...
%        /holdoutCVP.TestSize;
%    resubMCE(i) = crossval(classf,words_train(:,fs),gender_train,'partition',resubCVP)/...
%        resubCVP.TestSize;
% end
%  plot(nfs, testMCE,'o',nfs,resubMCE,'r^');
%  xlabel('Number of Features');
%  ylabel('MCE');
%  legend({'MCE on the test set' 'Resubstitution MCE'},'location','NW');
%  title('Simple Filter Feature Selection Method');



tenfoldCVP=cvpartition(gender_train_train,'kfold',10);

fs1 = featureIdxSortbyP(1:3000);
fsLocal = sequentialfs(classf,words_train_train(:,fs1),gender_train_train,'cv',tenfoldCVP);
fs1(fsLocal)

testMCELocal = crossval(classf,words_train(:,fs1(fsLocal)),gender_train,'partition',...
    holdoutCVP)/holdoutCVP.TestSize;

