% Here is the code for generate the feature template
clear
clc
close all

% there are 3 classes 
template_class1 = imread('class1.tiff','tiff');
template_class2 = imread('class2.tiff','tiff');
template_class3 = imread('class3.tiff','tiff');

template_class1 = double(template_class1(:,:,2))/255;
template_class2 = double(template_class2(:,:,2))/255;
template_class3 = double(template_class3(:,:,2))/255;

figure; 
subplot(1,3,1); imagesc(template_class1); daspect([1 1 1]); title('template1');
subplot(1,3,2); imagesc(template_class2); daspect([1 1 1]); title('template2');
subplot(1,3,3); imagesc(template_class3); daspect([1 1 1]); title('template3');

% Make observations from the template by adding noise
N = 100;
ns_lev = 2;
for i = 1:N
    class1(:,:,i) = template_class1 + ns_lev*randn(size(template_class1));
    class2(:,:,i) = template_class2 + ns_lev*randn(size(template_class2));
    class3(:,:,i) = template_class3 + ns_lev*randn(size(template_class3));
end

i = 20;
figure; 
subplot(1,3,1); imagesc(class1(:,:,i)); daspect([1 1 1]); title('template1'); caxis([-1 2]);
subplot(1,3,2); imagesc(class2(:,:,i)); daspect([1 1 1]); title('template2'); caxis([-1 2]);
subplot(1,3,3); imagesc(class3(:,:,i)); daspect([1 1 1]); title('template3'); caxis([-1 2]);

save('data_block_c3_2',...
    'ns_lev',...
    'template_class1','class1',...
    'template_class2','class2',...
    'template_class3','class3'...
    )
