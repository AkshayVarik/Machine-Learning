clear
clc
load('E:\Machine Learning\Final Project\kit_latest\kit\train\train.mat');
load('E:\Machine Learning\Final Project\kit_latest\kit\test\test.mat');
load('E:\Machine Learning\Final Project\kit\train\unknown_index.mat');
load('E:\Machine Learning\Final Project\kit\train\chi_squared_sorted_indices.mat'); 
load('E:\Machine Learning\Final Project\kit\train\parition_idx.mat'); 

% net_training_sum=sum(words_train);
% net_training_zero=find(~net_training_sum);
% net_training_nonzero=find(net_training_sum);
% words_train=words_train(:,net_training_nonzero);
% words_test=words_test(:,net_training_nonzero);

data = words_train;
ratio=0.75;
numInst = size(data,1);
numtrain = floor(ratio*numInst);

fr=chi_squared_sorted_indices;
fr=fr(3755:end,:);
data=data(:,fr);

words_train_train=data(partition_idx(1:numtrain,:));
words_train_test=data(partition_idx(numtrain+1:end,:));
genders_train_train=gender_train(partition_idx(1:numtrain)
genders_train_test=gender_train(partition_idx(numtrain+1:end));


% data = words_train;
% labels = gender_train;
% ratio = 0.75;
label_matrix_matrix=[];
accuracy_all=[];

% for j=1:20
    j=1;
%     idx = randperm(size(data,1));
%     numInst = size(data,1);
%     numTrain = floor(ratio*numInst); numVal = numInst - numTrain;
%     words_train_train = data(idx(1:numTrain),:); 
%     words_train_test = data(idx(numTrain+1:end),:);    
%     genders_train_train = labels(idx(1:numTrain)); 
%     genders_train_test= labels(idx(numTrain+1:end));
%     image_features_train_train = image_features_train(idx(1:numTrain),:);
%     image_features_train_test = image_features_train(idx(numTrain+1:end),:);
    
%    [ranked,weights]=relieff(words_train_train,genders_train_train,10);
   

   k_start=2000;
   k_end=4998;
   step=100;
   k=k_start:step:k_end;
   label_matrix=[];
   
   for i=1:numel(k)       
    ranked_top=find(ranked<=k(i));
    words_train_train=words_train_train(:,ranked);
     c vv cx
    train_train_set=[words_train_train image_features_train_train];
    adaStump = fitensemble(train_train_set,genders_train_train,'AdaBoostM1',500,'Tree',...
    'Type','Classification');
    train_test_set=[words_train_test image_features_train_test];
    label=predict(adaStump,train_test_set);
    label_matrix=[label_matrix label];
        
    [r c]=size(train_test_set);
    accuracy(i) =(sum(genders_train_test == label_matrix(:,i)))/r;
    
   end
   
%    label_matrix_matrix=[label_matrix_matrix label_matrix];
%    accuracy_all=[accuracy_all accuracy];
% 
% end
