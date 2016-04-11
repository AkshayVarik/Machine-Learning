function tb_idx=filterTotalBoostPredicitons(score_tb,tb_thresh)
% ab_thresholds = [0.8 0.9 1.0 1.1 1.2 1.4 1.6 1.8 1.9 2.1 2.3 2.5 2.7];
% tb_thresholds = [0.01 0.015 0.02 0.025 0.03 0.04];
% tb_thresh = 0.03;
tb_idx = [];
for j=1:size(score_tb,1)
    if(nnz(score_tb(j,:)>tb_thresh)>0)
        tb_idx = [tb_idx;j];
    end
end
% mean(pred_tb(tb_idx)==validationLabels(tb_idx))
% nnz(pred_svm(tb_idx)-pred_tb(tb_idx))
end
% mean(pred_ab(ab_idx)==validationLabels(ab_idx))
% temp = find(pred_svm(ab_idx)~=pred_ab(ab_idx))
% temp = ab_idx(temp);