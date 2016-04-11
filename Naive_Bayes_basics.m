%Naive Bayes algorithm 

load('E:\Machine Learning\Final Project\kit_latest\kit\train\train.mat');
image_features_train_train=image_features_train(1:3000,:);
image_features_train_test=image_features_train(3000:4998,:);
gender_train_train=gender_train(1:3001,:);
gender_train_test=gender_train(3001:4998,:);

[r c]=size(image_features_train_test);

index_female=[];
index_male=[];
for i=1:3000
    if gender_train_train(i)==1
        index_female=[index_female i];
    else
        index_male=[index_male i];
    end
end

image_features_train_train_female=image_features_train_train(index_female,:);
image_features_train_train_male=image_features_train_train(index_male,:);

V_female=var(image_features_train_train_female);
V_male=var(image_features_train_train_male);

M_female=mean(image_features_train_train_female);
M_male=mean(image_features_train_train_male);
        
P_male=0.5;
P_female=0.5;
for i=1:r
    P_age_male(i)=(1/sqrt(2*pi*V_male(1)))*exp((-(image_features_train_test(i,1)-M_male(1))^2)/2*V_male(1));
    P_smile_male(i)=(1/sqrt(2*pi*V_male(2)))*exp((-(image_features_train_test(i,2)-M_male(2))^2)/2*V_male(2));
    P_glasses_male(i)=(1/sqrt(2*pi*V_male(3)))*exp((-(image_features_train_test(i,3)-M_male(3))^2)/2*V_male(3));
    P_FacePitch_male(i)=(1/sqrt(2*pi*V_male(4)))*exp((-(image_features_train_test(i,4)-M_male(4))^2)/2*V_male(4));
    P_FaceRoll_male(i)=(1/sqrt(2*pi*V_male(5)))*exp((-(image_features_train_test(i,5)-M_male(5))^2)/2*V_male(5));
    P_FaceYaw_male(i)=(1/sqrt(2*pi*V_male(6)))*exp((-(image_features_train_test(i,6)-M_male(6))^2)/2*V_male(6));
    P_FaceSize_male(i)=(1/sqrt(2*pi*V_male(7)))*exp((-(image_features_train_test(i,7)-M_male(7))^2)/2*V_male(7));
    
    P_age_female(i)=(1/sqrt(2*pi*V_female(1)))*exp((-(image_features_train_test(i,1)-M_female(1))^2)/2*V_female(1));
    P_smile_female(i)=(1/sqrt(2*pi*V_female(2)))*exp((-(image_features_train_test(i,2)-M_female(2))^2)/2*V_female(2));
    P_glasses_female(i)=(1/sqrt(2*pi*V_female(3)))*exp((-(image_features_train_test(i,3)-M_female(3))^2)/2*V_female(3));
    P_FacePitch_female(i)=(1/sqrt(2*pi*V_female(4)))*exp((-(image_features_train_test(i,4)-M_female(4))^2)/2*V_female(4));
    P_FaceRoll_female(i)=(1/sqrt(2*pi*V_female(5)))*exp((-(image_features_train_test(i,5)-M_female(5))^2)/2*V_female(5));
    P_FaceYaw_female(i)=(1/sqrt(2*pi*V_female(6)))*exp((-(image_features_train_test(i,6)-M_female(6))^2)/2*V_female(6));
    P_FaceSize_female(i)=(1/sqrt(2*pi*V_female(7)))*exp((-(image_features_train_test(i,7)-M_female(7))^2)/2*V_female(7));
    
    Evidence(i)=(P_male*P_age_male(i)*P_smile_male(i)*P_glasses_male(i)*P_FacePitch_male(i)*P_FaceRoll_male(i)*P_FaceYaw_male(i)*P_FaceSize_male(i))+(P_female*P_age_female(i)*P_smile_female(i)*P_glasses_female(i)*P_FacePitch_female(i)*P_FaceRoll_female(i)*P_FaceYaw_female(i)*P_FaceSize_female(i));
    Posterior_male(i)=(P_male*P_age_male(i)*P_smile_male(i)*P_glasses_male(i)*P_FacePitch_male(i)*P_FaceRoll_male(i)*P_FaceYaw_male(i)*P_FaceSize_male(i))/Evidence(i);
    Posterior_female(i)=(P_female*P_age_female(i)*P_smile_female(i)*P_glasses_female(i)*P_FacePitch_female(i)*P_FaceRoll_female(i)*P_FaceYaw_female(i)*P_FaceSize_female(i))/Evidence(i);

    if Posterior_male(i)<Posterior_female(i)
        predictlabel(i)=1;
    else
        predictlabel(i)=0;      
    end 
end
    predictlabel=predictlabel';
    accuracy =(sum(gender_train_test == predictlabel))/r