function predictions = make_final_prediction(model,X_test,X_train)
% Input
% X_test : a nxp vector representing "n" test samples with p features.
% X_test=[words images image_features] a n-by-35007 vector
% model : struct, what you initialized from init_model.m
%
% Output
% prediction : a nx1 which is your prediction of the test samples
addpath(genpath('./faceDetection/'));
addpath(genpath('./libsvm/'));
load('finalIndices.mat');
ttotal = tic;
trainingWords = X_train(:,1:5000);
trainingImageFeatures = X_train(:,35001:end);
trainingData = [trainingWords trainingImageFeatures];
trainingData = trainingData(:,final);
% trainingLabels = X_train(:,end);
testingWords = X_test(:,1:5000);
testingImages = X_test(:,5001:35000);
testingImageFeatures = X_test(:,35001:end);
testingData = [testingWords testingImageFeatures];
testingData = testingData(:,final);
testingLabels = zeros(size(testingData,1),1);
flag = zeros(size(testingLabels,1),1);
k = @(x,x2)kernel_intersection(x, x2);
disp('Predicting SVM');
tsvm = tic;
[pred_svm, score_svm] = svmtestPredict(double(trainingData),testingData, testingLabels, model.new_svm, k);
% [info] = kernel_libsvm(double(trainingData), trainingLabels, testingData, testingLabels, k);
toc(tsvm);
% pred_svm = info.yhat;
% score_svm = info.vals;
svm_thresh = 0.89;
svm_idx = filterPredictions(score_svm,svm_thresh);
flag(svm_idx) = 1;
disp('Predicting GentleBoost')
[pred_gb,score_gb] = predict(model.adaBoost,testingData);
disp('Predicting AdaBoost')
[pred_ab,score_ab] = predict(model.gentleBoost,testingData);
gb_thresh = 2.85;
ab_thresh = 5.75; %was 1.9 for 90.77
gb_idx = filterPredictions(score_gb,gb_thresh);
ab_idx = filterPredictions(score_ab,ab_thresh);
flag(gb_idx) = 1;
flag(ab_idx) = 1;
weak_idx = find(flag==0);
fr_testingImages = testingImages(weak_idx,:);
fr_thresh = 0.545;
disp('Checking Faces');
[pred_fr,fr_idx] = faceDetection(fr_testingImages,fr_thresh);
fr_to_org_mapping = weak_idx(fr_idx);
[predictions,indices_changed,indices_corrected_by] = finalPredictions(pred_svm,pred_gb,gb_idx,pred_ab,ab_idx,pred_fr,fr_to_org_mapping);
toc(ttotal)

% idx = indices_changed;%same_ab_tb;
% img_X = testingImages(idx,:);
% for i=1:size(img_X,1)
%   cur_idx = idx(i);
%   if indices_corrected_by(i)~=3 
%       i
%       indices_corrected_by(i)
%       cur_row=img_X(i,:);
%       cur_img=reshape(cur_row,[100 100 3]);
%       figure()
%       title_here = sprintf('SVM = %d ENS = %d Method=%d',pred_svm(cur_idx),predictions(cur_idx),indices_corrected_by(i));
%       title(title_here)
%     %   cur_idx
%     %   pred_ab(cur_idx)
%     %   pred_svm(cur_idx)
%       hold on
%       imshow(uint8(cur_img));
%       pause(3);
%       close all
%   end
% end
% % end
% % dlmwrite('submit.txt', new_pred);
% % Chat Conversation End

end
