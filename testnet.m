 %% Try to classify something else
 clear;clc

 net = coder.loadDeepLearningNetwork('model\densenet201_sgdm\Densenet201_sgdm_1.mat','net');

[filerootd, pathname1, filterindex1] = uigetfile({'*.png'}, ...
   'Select an image');
x=imresize(imread([pathname1, filerootd]),[224 224]);
[a,b,c]=size(x);
if c==1
  img=cat(2,x,x,x);
else
    img=x;
end
actualLabel='--';
[YPred,scores] = net.classify(img);
switch(YPred)
    case 'COVID'
        score=scores(1);
     case 'NORMAL'
      score=scores(2);
end
imshow(img);
title(['Predicted: ' char(YPred) mat2str(floor(score*100)) '%',' Actual: ' char(actualLabel)])



