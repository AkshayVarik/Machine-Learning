clear
clc
load('E:\Machine Learning\Final Project\kit_latest\kit\train\train.mat');
load('E:\Machine Learning\Final Project\kit_latest\kit\test\test.mat');

net_training_sum=sum(words_train);
net_training_zero=find(~net_training_sum);
net_training_nonzero=find(net_training_sum);
words_train=words_train(:,net_training_nonzero);
words_test=words_test(:,net_training_nonzero);

words_train_train=words_train(1:4000,:);
words_train_test=words_train(4001:4998,:);
genders_train_train=gender_train(1:4000,:);
genders_train_test=gender_train(4001:4998,:);

[r c]=size(words_train_test);

lda=fitcdiscr(words_train_train, genders_train_train);
ldaClass=resubPredict(lda);
ldaResubErr=resubLoss(lda);
[ldaResubCM, grpOrder]=confusionmat(genders_train_train, ldaClass);

label=predict(lda,words_train_test);
accuracy =(sum(genders_train_test == label))/r
