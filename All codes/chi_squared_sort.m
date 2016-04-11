clear
clc
load('E:\Machine Learning\Final Project\kit\train\Chi_squared_all_training_and_image_features.mat');

chi_squared_sorted=sort(ans);
[r c]=size(chi_squared_sorted);
[m n]=size(ans);

chi_squared_sorted_indices=[];
for i=1:r
    for j=1:m
        if chi_squared_sorted(i)==ans(j)
            chi_squared_sorted_indices=[chi_squared_sorted_indices j];
        end
    end
end

 chi_squared_sorted_indices= chi_squared_sorted_indices';
 [u v]=size(chi_squared_sorted_indices);
 s=u-r;
 t=s+1;
 chi_squared_sorted_indices=chi_squared_sorted_indices(t:end,:);
 
 if length(chi_squared_sorted_indices) == length(unique(chi_squared_sorted_indices))
    pass=1
 else
     fail=1
 end