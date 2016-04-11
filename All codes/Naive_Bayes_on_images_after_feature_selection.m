load('E:\Machine Learning\Final Project\kit_latest\kit\train\train.mat');

insight=sum(images_train(:,:)==0);
index=[];
for i=1:30000
    if insight(i)<1000
        index=[index i];
    end 
end 
images_train_reduced=images_train(:,index);
% [row col]=size(words_train_reduced);

images_train_reduced_train=images_train_reduced(1:4000,:);
images_train_reduced_test=images_train_reduced(4000:4998,:);
gender_train_train=gender_train(1:4000,:);
gender_train_test=gender_train(4000:4998,:);

[r c]=size(images_train_reduced_test);

% [coeff,score,latent] = pca(images_train_train);
% projections=images_train_train*coeff;
% projections_rd=projections(:,1:500);
% images_train_test=images_train_test(:,1:500);

mdl=fitcnb(images_train_reduced_train,gender_train_train,'Distribution','mn');
predictlabel=predict(mdl,images_train_reduced_test);
accuracy =(sum(gender_train_test == predictlabel))/r