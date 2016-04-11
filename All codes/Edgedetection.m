% % load('C:\Users\Akshay\Pictures');
% I = imread('AV.png');
% Img=rgb2gray(I);
% imshow(Img);
% 
% BW1 = edge(Img,'sobel');
% BW2 = edge(Img,'canny');
% figure;
% imshowpair(BW1,BW2,'montage')
% title('Sobel Filter                         Canny Filter');





clear all
close all
load('E:\Machine Learning\Final Project\kit_latest\kit\train\train.mat');
img_X = images_train;
for i=1:size(img_X,1)
  cur_row=img_X(i,:);
  cur_img=reshape(cur_row,[100 100 3]);
  figure()
  
Img=rgb2gray(img_X);
imshow(Img);

BW1 = edge(Img,'sobel');
BW2 = edge(Img,'canny');
figure;
imshowpair(BW1,BW2,'montage')
  
  title(gender_train(i))
  hold on
  imshow(uint8(cur_img));
  pause(3);
  close all
end

