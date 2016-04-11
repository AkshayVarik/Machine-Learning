clear
clc
load('E:\Machine Learning\Final Project\kit_latest\kit\train\train.mat');
load('E:\Machine Learning\Final Project\kit_latest\kit\test\test.mat');

% [images_train, gender_train]=imagetrain_dataset;

% clf
% for i=1:20
%     subplot(4,5,i);
%     imshow(images_train{i});
% end

% img_X = images_train;
% for i=1:size(img_X,1)
%   cur_row=img_X(i,:);
%   cur_img=reshape(cur_row,[100 100 3]);
%   figure()
% %   title(predictlabel(i))
%   hold on
%   imshow(uint8(cur_img));
%   pause(3);
%   close all
% end


rng('default');
hiddenSize1=100;
autoenc1=trainAutoencoder(images_train,hiddenSize1,...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);
view(autoenc1)
plotWeights(autoenc1);
feat1 = encode(autoenc1,images_train);


hiddenSize2 = 50;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);
feat2 = encode(autoenc2,feat1);


softnet = trainSoftmaxLayer(feat2,gender_train,'MaxEpochs',400);
view(softnet)
view(autoenc1)
view(autoenc2)
view(softnet)


deepnet = stack(autoenc1,autoenc2,softnet);
view(deepnet)
