load('E:\Machine Learning\Final Project\kit_latest\kit\train\train.mat');
image_features_train_train=image_features_train(1:3000,:);
image_features_train_test=image_features_train(3000:4998,:);
gender_train_train=gender_train(1:3000,:);
gender_train_test=gender_train(3000:4998,:);

[r c]=size(image_features_train_test);

mdl=fitcnb(image_features_train_train,gender_train_train,'Distribution','mvmn');
predictlabel=predict(mdl,image_features_train_test);
accuracy =(sum(gender_train_test == predictlabel))/r