function model = init_model()
addpath(genpath('./libsvm/'));
addpath(genpath('./faceDetection/'));
addpath('./models/');
load('./models/my_svm_model.mat');
load('./models/ensemble_ab.mat');
load('./models/ensemble_gb.mat');
load('finalIndices.mat');
model.new_svm = info.model;
model.adaBoost = ensemble_ab;
model.gentleBoost = ensemble_gb;
end

% Example:
% model.svmw = SVM.w;
% model.lrw = LR.w;
% model.classifier_weight = [c_SVM, c_LR];
