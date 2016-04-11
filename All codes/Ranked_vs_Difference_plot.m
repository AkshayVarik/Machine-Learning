clear
clc
load('E:\Machine Learning\Final Project\kit\train\difference.mat');
load('E:\Machine Learning\Final Project\kit\train\Ranked.mat');

ranked_sorted=sort(ranked);
Difference_sorted=Difference(ranked_sorted);

plot(ranked_sorted,Difference_sorted)
xlabel('Rank');
ylabel('Difference');