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

% [r c]=size(words_train_test);
% [r c]=size(words_test);

[ranked,weights]=relieff(words_train_train,genders_train_train,10);

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

Female_indices=find(genders_train_train);
Male_indices=find(~genders_train_train);

words_train_train_male=words_train_train(Male_indices,:);
words_train_train_female=words_train_train(Female_indices,:);

words_train_train_male_count=sum(words_train_train_male);
words_train_train_female_count=sum(words_train_train_female);

% Difference=words_train_train_male_count-words_train_train_female_count;
Difference(j,:)=words_train_train_male_count-words_train_train_female_count;
% [row col]=size(Difference);
% a=[];
% for i=1:col
%     if Difference(i)<0
%         a=[a i];
%     end
% end
% Max_Difference=max(Difference);
% Min_Difference=min(Difference);
% Max_Difference_index=find(Difference==-673306);
% Min_Difference_index=find(Difference==8134);

% Mean=mean(Difference);
% Variance=var(Difference);
% Standard_Deviation=std(Difference);
% norm=normpdf(Difference,Mean,Standard_Deviation);
% bar(Difference)

% Keep_word_indices=find(Difference>=0 | Difference<0);
% Keep_word_indices=find(Difference(j,:)>10 | Difference(j,:)<-10);
% Removed_word_indices=find(Difference==0);
% Keep_word_train=words_train_train(:,Keep_word_indices);
% Keep_word_test=words_train_test(:,Keep_word_indices);
% words_test=words_test(:,Keep_word_indices);


% Keep_word_train=words_train_train(:,net_training_nonzero);
% Keep_word_test=words_train_test(:,net_training_nonzero);% 
% words_test=words_test(:,net_training_nonzero);

end 
Standard_Deviation=std(Difference,0,2);
Max_standard_deviation=max(Standard_Deviation);
Max_standard_deviation_index=find(Max_standard_deviation);
Chosen_Difference=Difference(Max_standard_deviation_index,:); %Chosen row of difference with maximum standard deviation

Keep_word_indices=find(Chosen_Difference>0 | Chosen_Difference<0);
Removed_word_indices=find(Chosen_Difference==0);
Keep_word_train=words_train_train(:,Keep_word_indices);
Keep_word_test=words_train_test(:,Keep_word_indices);
words_test=words_test(:,Keep_word_indices);
