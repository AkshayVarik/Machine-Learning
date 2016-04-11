load('E:\Machine Learning\Final Project\kit_latest\kit\train\train.mat');
words_train_train=words_train(1:4000,:);
words_train_test=words_train(4000:4998,:);
images_train_train=images_train(1:4000,:);
images_train_test=images_train(4000:4998,:);
gender_train_train=gender_train(1:4000,:);
gender_train_test=gender_train(4000:4998,:);

net_train_train=[images_train_train, words_train_train];
net_train_test=[images_train_test, words_train_test];

[r c]=size(net_train_test);

% [coeff,score,latent] = pca(words_train_train);
% projections=words_train_train*coeff;
% projections_rd=projections(:,1:50);
% words_train_test=words_train_test(:,1:50);

mdl=fitcnb(net_train_train,gender_train_train,'Distribution','mn');
predictlabel=predict(mdl,net_train_test);
accuracy =(sum(gender_train_test == predictlabel))/r