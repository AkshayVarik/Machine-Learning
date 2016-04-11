% classification
clear
close all
clc

%% ==== make the data in a proper format #obs x #feature
filename = 'data_block_c3_2';
load(filename);

num_class = 3;
[num_row, num_col, m] = size(class1);
n = num_row*num_col;
X = size(m,n); 

hndl = figure; 
subplot(1,3,1); imagesc(template_class1); 
daspect([1 1 1]);
title('template of class1'); colorbar;
subplot(1,3,2); imagesc(template_class2); 
daspect([1 1 1]);
title('template of class2'); colorbar;
subplot(1,3,3); imagesc(template_class3); 
daspect([1 1 1]);
title('template of class3'); colorbar;
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
print(hndl, '-djpeg', ['template_',filename,'.jpg']);

hndl = figure; 
subplot(1,3,1); imagesc(class1(:,:,1)); 
daspect([1 1 1]);
title('example of class1'); colorbar;
subplot(1,3,2); imagesc(class2(:,:,1)); 
daspect([1 1 1]);
title('example of class2'); colorbar;
subplot(1,3,3); imagesc(class3(:,:,1)); 
daspect([1 1 1]);
title('example of class3'); colorbar;
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
print(hndl, '-djpeg', ['example_',filename,'.jpg']);

c1 = zeros(m,n);
for i = 1:m
    tmp = class1(:,:,i);
    c1(i,:) = tmp(:)';
end

c2 = zeros(m,n);
for i = 1:m
    tmp = class2(:,:,i);
    c2(i,:) = tmp(:)';
end

c3 = zeros(m,n);
for i = 1:m
    tmp = class3(:,:,i);
    c3(i,:) = tmp(:)';
end

num_c1 = size(c1,1);
num_c2 = size(c2,1);
num_c3 = size(c3,1);


X = [c1;c2;c3];
class_vector = [1+zeros(num_c1,1);
    2+zeros(num_c2,1)
    3+zeros(num_c3,1)];

clear class1 class2 class3 c1 c2 c3

figure;
subplot(1,2,1); imagesc(class_vector);
subplot(1,2,2); imagesc(X);

%% ===== separate the data into train and test set
idx = randn(length(class_vector),1);
idx_thr = 0.8;
X_train = X(idx<=idx_thr,:);
c_train = class_vector(idx<=idx_thr);
m_train = sum(idx<=idx_thr);
X_train = zscore(X_train,0,1);


X_test = X(idx>idx_thr,:);
c_test = class_vector(idx>idx_thr);
m_test = sum(idx>idx_thr);
X_test = zscore(X_test,0,1);

figure;
subplot(2,2,1); imagesc(c_train);
subplot(2,2,2); imagesc(X_train);
subplot(2,2,3); imagesc(c_test);
subplot(2,2,4); imagesc(X_test);

% === classification ====
addpath('../toolbox_logistic_regression_coursera/');
addpath('../toolbox_misc');

% Train the multi-class LR
lambda = 0.1;
[all_theta] = oneVsAll(X_train,c_train, num_class, lambda);
% Test on the test set
[c_hat, ~, post] = predictOneVsAll(all_theta, X_test);
accuracy = sum(c_hat==c_test)/m_test;

w = log(all_theta(:,1:end-1).^2);
w_max = max(w(:));
w_min = min(w(:));
w_reshaped(:,:,1) = reshape(w(1,:),num_row,num_col);
w_reshaped(:,:,2) = reshape(w(2,:),num_row,num_col);
w_reshaped(:,:,3) = reshape(w(3,:),num_row,num_col);


%% ==== compare to the mi score
addpath('../toolbox_ITL/');
kernel_width = 0.05; 
numEstimate = 100;
mi_score = mi_sample(c_train,X_train,kernel_width,numEstimate);
mi_reshaped = reshape(mi_score,num_row,num_col);

%% === save the value ===
save(['result_',filename],...
    'accuracy',...
    'w_reshaped',...
    'mi_reshaped'...
    );

%% == plot logistic regression weight

% figure; 
% imagesc(w);

hndl = figure; 
subplot(2,2,1); imagesc(w_reshaped(:,:,1)); 
caxis([w_min w_max]); daspect([1 1 1]);
title('weight for class1'); colorbar;
subplot(2,2,2); imagesc(w_reshaped(:,:,2)); 
caxis([w_min w_max]); daspect([1 1 1]);
title('weight for class2'); colorbar;
subplot(2,2,3); imagesc(w_reshaped(:,:,3)); 
caxis([w_min w_max]); daspect([1 1 1]);
title('weight for class3'); colorbar;
subplot(2,2,4); imagesc(mi_reshaped); 
daspect([1 1 1]);
title('mutual information'); colorbar;

set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
print(hndl, '-djpeg', ['weight_',filename,'.jpg']);

%% plot the histogram of the weights
ww = w_reshaped(:,:,1);
figure; hist(ww(:),100);
w_sorted = sort(ww(:),'ascend');
Fw = (1:length(w_sorted))/length(w_sorted);
hndl = figure; semilogy(w_sorted, Fw,'b-','LineWidth',3); ylim([-0.5 1.5]);
xlabel('weight for each feature');
ylabel('cumultive density');
print(hndl, '-djpeg', ['weight_cumulative_',filename,'.jpg']);

%%
thrsld = -11;
selected_w = ww;
selected_w(ww >= thrsld) = 1;
selected_w(ww < thrsld) = 0;
hndl = figure; imagesc(selected_w); colorbar;
title(['selected feature using threshold=',num2str(thrsld)]);
print(hndl, '-djpeg', ['selected_feature_thr',num2str(thrsld),'_',filename,'.jpg']);