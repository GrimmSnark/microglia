function trainUNet4MicrogliaMaskExtraction(dataDir, saveLoc, viewTestImages)

%% defaults

if nargin < 2 || isempty(saveLoc)
    saveLoc = [dataDir '\Nets\'];
end

if nargin < 3 || isempty(viewTestImages)
    viewTestImages = 0;
end

maskLocation = [dataDir '\masks\'];
imageLocation = [dataDir '\images\'];
patchSize = 256;
testImageNo = 8;

%% load data
imageData = imageDatastore(imageLocation);
imageData.ReadSize = 1;

%% get the ROI masks
labeledROIDataStore = pixelLabelDatastore(maskLocation, {'Background', 'Cell'}, [0 255]);

% extract 'patches' gets the data into the right format for processing
dsTrain = randomPatchExtractionDatastore(imageData, labeledROIDataStore, patchSize ,'PatchesPerImage', 20);
dsTrain.MiniBatchSize = 10;

% subset for validation (I know this is double dipping)
num2Validate = 100;
dsValidate = partition(dsTrain,num2Validate,1);

%examine dsTrain
if viewTestImages == 1
    minibatch = preview(dsTrain);
    inputs = minibatch.InputImage;
    responses = minibatch.ResponsePixelLabelImage;
    test = cat(2,inputs,responses);

    for x = 1:testImageNo
        C(:,:,:,x) = labeloverlay(mat2gray(test{x,1}),test{x,2},'Transparency',0.8);
    end

    montage(C);
    pause
end

%% build neural net
inputTileSize = [patchSize patchSize]; % based on the patch extraction size
numClasses = 2;
lgraph = unetLayersV2(inputTileSize, numClasses, 'EncoderDepth', 6);

% taken from example 3D segementation MRI
outputLayer = dicePixelClassificationLayer('Name','Dice Layer Output');
lgraph = replaceLayer(lgraph,'Segmentation-Layer',outputLayer);

disp(lgraph.Layers)


% train options
initialLearningRate = 0.01;
maxEpochs = 100;
minibatchSize = 16;
l2reg = 0.001;

if ~exist([saveLoc '\checkpoint\'])
    mkdir([saveLoc '\checkpoint\']);
end

options = trainingOptions('rmsprop',...
    'ExecutionEnvironment', 'auto', ...
    'InitialLearnRate',initialLearningRate, ...
    'L2Regularization',l2reg,...
    'MaxEpochs',maxEpochs,...
    'MiniBatchSize',minibatchSize,...
    'ValidationData',dsValidate, ...
    'ValidationFrequency',100, ...
    'LearnRateSchedule','piecewise',...
    'Shuffle','every-epoch',...
    'CheckpointPath',[saveLoc '\checkpoint\'], ...
    'GradientThresholdMethod','l2norm',...
    'GradientThreshold',0.05, ...
    'Plots','training-progress', ...
    'VerboseFrequency',50);


%% run the net
modelDateTime = datestr(now,'dd-mmm-yyyy-HH-MM-SS');
[net,info] = trainNetwork(dsTrain,lgraph,options);
save([saveLoc '\microgliaUNet_Patch256-' modelDateTime '-Epoch-' num2str(maxEpochs) '.mat'],'net','options');

end

