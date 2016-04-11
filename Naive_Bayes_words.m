function predictions =Naive_Bayes_words(train_filename, test_filename)

load(train_filename);
load(test_filename);

net_training_sum=sum(words_train);
net_training_zero=find(~net_training_sum);
net_training_nonzero=find(net_training_sum);
words_train=words_train(:,net_training_nonzero);
words_test=words_test(:,net_training_nonzero);

words_train_train=words_train(1:4000,:);
words_train_test=words_train(4001:4998,:);
genders_train_train=gender_train(1:4000,:);
genders_train_test=gender_train(4001:4998,:);

mdl=fitcnb(words_train_train,genders_train_train,'Distribution','mn');
CVMdl=crossval(mdl);
a=kfoldLoss(CVMdl);
CMdl=CVMdl.Trained{7};
[predictions,probability,cost]=predict(CMdl,words_test);

% [r c]=size(words_test);
% accuracy_nb =(sum(genders_train_test == predictions))/r

% tb_thresh=0.95;
% tb_idx=[];
% for j=1:size(probability,1)
%     if(nnz(probability(j,:)>tb_thresh)>0)
%         tb_idx = [tb_idx;j];
%     end
% end
    
