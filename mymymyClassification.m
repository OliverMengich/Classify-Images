 outputFolder = fullfile('Caltech101');
rootfolder = fullfile(outputFolder,'101_ObjectCategories');
categories = {'car_side','ceiling_fan','cup','chair','Faces'};
imds = imageDatastore(rootfolder,'IncludeSubfolders',true,'LabelSource','foldernames');
 [imdsTrainSet,imdsTestSet] = splitEachLabel(imds,0.7,'randomized');
% tbl = countEachLabel(imds);
%  minSetCount = min(tbl{:,2});
%  imds= splitEachLabel(imds,minSetCount,'randomize');
% countEachLabel(imds);
%  car_side = find(imds.Labels == 'car_side',1);
%  ceiling_fan = find(imds.Labels == 'ceiling_fan',1);
%  cup = find(imds.Labels == 'chair',1);
% Faces = find(imds.Labels == 'Faces',1);
 numTrainImages = numel(imdsTrainSet.Labels);
 idx = randperm(numTrainImages,16);
 figure
for i = 1:16
subplot(4,4,i)
  I = readimage(imdsTrainSet,idx(i));
   imshow(I)
end
net = alexnet;
net.Layers;
inputSize = net.Layers(1).InputSize;
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrainSet,'ColorPreprocessing','gray2rgb');
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTestSet,'ColorPreprocessing','gray2rgb');
layer = 'fc7';
featuresTrain = activations(net,augimdsTrain,'fc7','OutputAs','rows');
featuresTest = activations(net,augimdsTest,'fc7','OutputAs','rows');
YTrain = imdsTrainSet.Labels;
YTest = imdsTestSet.Labels;
classifier= fitcecoc(featuresTrain,YTrain);
 YPred = predict(classifier,featuresTest);

%  camera = webcam;
%  camera1 = camera.snapshot;
% camera1 = imresize(camera1,[227 227]);
% % label = classify(YPred,featuresTest);
% label = predict(classifier,camera1);
% imshow(camera1);
% title(char(label));
%    idx = [3,4,5,6,7];
% for i = 1:numel(idx)
% %     figure
%     subplot(2,3,i);
%     I = readimage(imdsTestSet,idx(i));
%     label = YPred(idx(i));
%     imshow(I);
%     title({char(label),num2str(max(score),2)});
% end
