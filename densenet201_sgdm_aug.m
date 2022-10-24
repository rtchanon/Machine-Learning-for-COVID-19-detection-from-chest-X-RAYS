clc;close all;clear;

digitDatasetPath = fullfile('COVID-19_Radiography_Dataset/');
 imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

countEachLabel(imds)

[imds,imdsTest] = splitEachLabel(imds,0.8,'randomized');
[imd1 imd2 imd3 imd4 imd5] = splitEachLabel(imds,0.20,0.20,0.20,0.20,0.20);
partStores{1} = imd1.Files;
partStores{2} = imd2.Files;
partStores{3} = imd3.Files;
partStores{4} = imd4.Files;
partStores{5} = imd5.Files;
k = 5;
idx = crossvalind('Kfold', k, k);   %crossvalind generates cross-validation indices

for foldi = 1:k
    test_idx = (idx == foldi);
    train_idx = ~test_idx;
    imdsValidation = imageDatastore(partStores{test_idx}, 'IncludeSubfolders', true,'LabelSource', 'foldernames');
    imdsTrain = imageDatastore(cat(1, partStores{train_idx}), 'IncludeSubfolders', true,'LabelSource', 'foldernames'); 

    net=densenet201;
    lgraph = layerGraph(net);
    clear net;
    
    numClasses = numel(categories(imdsTrain.Labels));
    
    newFCLayer = fullyConnectedLayer(numClasses,'Name','new_fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
    lgraph = replaceLayer(lgraph,'fc1000',newFCLayer);
    newClassLayer = softmaxLayer('Name','new_softmax');
    lgraph = replaceLayer(lgraph,'fc1000_softmax',newClassLayer);
    newClassLayer = classificationLayer('Name','new_classoutput');
    lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',newClassLayer);
    
    options = trainingOptions('sgdm',...
            'MaxEpochs',5,'MiniBatchSize',64,...
            'InitialLearnRate', 0.001, ...
            'Shuffle','every-epoch', ...
            'Momentum', 0.9, ...
            'ValidationData', imdsValidation, ...
            'ValidationFrequency', 20, ...
            'ExecutionEnvironment','gpu', ...
            'Verbose',false, ...
            'Plots','training-progress');
    
    augmenter = imageDataAugmenter( ...
            'RandRotation',[-5 5],'RandXShear',[-0.05 0.05],'RandYShear',[-0.05 0.05]);
    
    auimdsTrain = augmentedImageDatastore([224 224],imdsTrain,'DataAugmentation',augmenter);
    
    [netTransfer, info] = trainNetwork(auimdsTrain,lgraph,options);
    
    predicted_labels = classify(netTransfer,imdsTest);
    
    actual_labels=imdsTest.Labels;
    
    save(sprintf('Densenet201_sgdm_aug_%d',foldi),'netTransfer','test_idx','train_idx', 'info');
    
    % Confusion Matrix
    figure;
    plotconfusion(actual_labels,predicted_labels)
    title(sprintf('Confusion Matrix: Densenet201 sgdm aug_%d',foldi));

end
