%unetLayers Create U-Net for semantic segmentation using deep learning.
%   
%   MODIFIED WITH ADDED BATCH NORMALIZATION LAYERS
%   M Savage 20191220
%
%   U-Net is a convolutional neural network for semantic image
%   segmentation. It uses a pixelClassificationLayer to predict the
%   categorical label for every pixel in an input image. The network gets
%   its name from the "U" shape created when the layers are arranged in
%   order.
%
%   Use unetLayers to create the network architecture for U-Net. This
%   network must be trained using trainNetwork from Deep Learning Toolbox
%   before it can be used for semantic segmentation.
%
%   lgraph = unetLayers(imageSize, numClasses) returns U-Net layers
%   configured using the following inputs:
%
%   Inputs
%   ------
%   imageSize    - size of the network input image specified as a vector
%                  [H W] or [H W C], where H and W are the image height and
%                  width, and C is the number of image channels.
%
%   numClasses   - number of classes the network should be configured to
%                  classify.
%
%   [...] = unetLayers(imageSize, numClasses, Name, Value,___)
%   specifies additional name-value pair arguments described below:
%
%   'EncoderDepth'           U-Net is composed of an encoder sub-network
%                            and a corresponding decoder sub-network.
%                            Specify the depth of these networks as a
%                            scalar D. The depth of these networks
%                            determines the number of times an input image
%                            is downsampled or upsampled as it is
%                            processed. The encoder network downsamples the
%                            input image by a factor of 2^D. The decoder
%                            network performs the opposite operation and
%                            upsamples the encoder network output by a
%                            factor of 2^D. Typical depth of the encoder
%                            sub-network is 4.
%
%                            Default: 4
%
%   'NumOutputChannels'      Specify the number of output channels for the
%                            first encoder subsection. Each of the
%                            subsequent encoder subsections double the
%                            number of output channels. The number of
%                            output channels in the decoder sections is
%                            automatically set to match the corresponding
%                            encoder section.
%
%                            Default: 64
%
%   'FilterSize'             Specify the height and width used for all
%                            convolutional layer filters as a scalar or
%                            vector [H W]. When the size is a scalar, the
%                            same value is used for all layers. Typical
%                            values are between 3 and 7. The value must
%                            be odd.
%
%                            Default: 3
%
% Notes
% -----
% - This version of U-Net uses "same" padding for the convolutional layers
%   to enable a broader set of input image sizes. The original version did
%   not use padding and is constrained to a small set of input image sizes.
%
% - The sections within the U-Net encoder sub-networks are
%   made up of two sets of convolutional and ReLU layers followed by a 2x2
%   max-pooling layer. While the sections of the decoder network are made
%   up a transposed convolution layer (for upsampling) followed by two sets
%   of convolutional and ReLU layers.
%
% - Convolution layer weights in the encoder and decoder sub-networks are
%   initialized using the 'He' weight initialization method. All bias terms
%   are initialized to zero.
%
% - Each encoder section has a 2x2 max-pooling layer which halves the image
%   size. The input image size should be such that the size of the image
%   before each of the max-pooling operation is even in the first two
%   dimensions. Otherwise, there would be an image size mismatch between
%   the encoder and the decoder sub-networks resulting in an error.
%
%   Example 1 - Create U-Net with custom encoder/decoder depth.
%   ------------------------------------------------------------
%   % Create U-Net layers with an encoder/decoder depth of 3.
%   imageSize = [480 640 3];
%   numClasses = 5;
%   encoderDepth = 3;
%   lgraph = unetLayers(imageSize, numClasses, 'EncoderDepth', encoderDepth)
%
%   % Display network.
%   figure
%   plot(lgraph)
%
%   Example 2 - Train U-Net.
%   -------------------------
%   % Load training images and pixel labels.
%   dataSetDir = fullfile(toolboxdir('vision'),'visiondata','triangleImages');
%   imageDir = fullfile(dataSetDir, 'trainingImages');
%   labelDir = fullfile(dataSetDir, 'trainingLabels');
%
%   % Create an imageDatastore holding the training images.
%   imds = imageDatastore(imageDir);
%
%   % Define the class names and their associated label IDs.
%   classNames = ["triangle", "background"];
%   labelIDs   = [255 0];
%
%   % Create a pixelLabelDatastore holding the ground truth pixel labels for
%   % the training images.
%   pxds = pixelLabelDatastore(labelDir, classNames, labelIDs);
%
%   % Create U-Net.
%   imageSize = [32 32];
%   numClasses = 2;
%   lgraph = unetLayers(imageSize, numClasses)
%
%   % Combine image and pixel label data to train a semantic segmentation network.
%   ds = pixelLabelImageDatastore(imds,pxds);
%
%   % Setup training options.
%   options = trainingOptions('sgdm', 'InitialLearnRate', 1e-3, ...
%       'MaxEpochs', 20, 'VerboseFrequency', 10);
%
%   % Train network.
%   net = trainNetwork(ds, lgraph, options)
%
% See also segnetLayers, fcnLayers, vgg16, vgg19, pixelClassificationLayer,
%          LayerGraph, trainNetwork, DAGNetwork, semanticseg,
%          pixelLabelImageDatastore.

% References
% ----------
%
% [1] Olaf Ronneberger, Philipp Fischer, and Thomas Brox, U-Net:
%     Convolutional Networks for Biomedical Image Segmentation, Medical
%     Image Computing and Computer-Assisted Intervention (MICCAI),
%     Springer, LNCS, Vol.9351: 234--241, 2015, available at
%     arXiv:1505.04597
%
% [2] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing
%     human-level performance on imagenet classification." Proceedings of the
%     IEEE international conference on computer vision. 2015.

% Copyright 2017-2018 The MathWorks, Inc.

function lgraph = unetLayersV2(imageSize, numClasses, varargin)

vision.internal.requiresNeuralToolbox(mfilename);

narginchk(2,8);

args = iParseInputs(imageSize, numClasses, varargin{:});

if numel(args.imageSize) == 3
    inputNumChannels = args.imageSize(3);
else
    inputNumChannels = 1;
end

encoderDepth = args.EncoderDepth;
initialEncoderNumChannels = args.NumOutputChannels;
inputTileSize = args.imageSize;
convFilterSize = args.FilterSize;

inputlayer = imageInputLayer(inputTileSize,'Name','ImageInputLayer');

[encoder, finalNumChannels] = iCreateEncoder(encoderDepth, convFilterSize, initialEncoderNumChannels, inputNumChannels);

firstConv = createAndInitializeConvLayer(convFilterSize, finalNumChannels, ...
    2*finalNumChannels, 'Bridge-Conv-1');

firstBatchNorm = batchNormalizationLayer('Name', 'Bridge-BatchNorm-1');

firstReLU = reluLayer('Name','Bridge-ReLU-1');

secondConv = createAndInitializeConvLayer(convFilterSize, 2*finalNumChannels, ...
    2*finalNumChannels, 'Bridge-Conv-2');

secondBatchNorm = batchNormalizationLayer('Name', 'Bridge-BatchNorm-2');

secondReLU = reluLayer('Name','Bridge-ReLU-2');

encoderDecoderBridge = [firstConv; firstBatchNorm ;  firstReLU; secondConv; secondBatchNorm ; secondReLU];

dropOutLayer = dropoutLayer(0.5,'Name','Bridge-DropOut');
encoderDecoderBridge = [encoderDecoderBridge; dropOutLayer];

initialDecoderNumChannels = finalNumChannels;

upConvFilterSize = 2;

[decoder, finalDecoderNumChannels] = iCreateDecoder(encoderDepth, upConvFilterSize, convFilterSize, initialDecoderNumChannels);

layers = [inputlayer; encoder; encoderDecoderBridge; decoder];

numClasses = args.numClasses;
finalConv = convolution2dLayer(1,numClasses,...
    'BiasL2Factor',0,...
    'Name','Final-ConvolutionLayer');
finalConv.Weights = randn(1,1,finalDecoderNumChannels,numClasses);
finalConv.Bias = zeros(1,1,numClasses);

smLayer = softmaxLayer('Name','Softmax-Layer');
pixelClassLayer = pixelClassificationLayer('Name','Segmentation-Layer');

layers = [layers; finalConv; smLayer; pixelClassLayer];

lgraph = layerGraph(layers);

for depth = 1:encoderDepth
    startLayer = sprintf('Encoder-Stage-%d-ReLU-2',depth);
    endLayer = sprintf('Decoder-Stage-%d-DepthConcatenation/in2',encoderDepth-depth + 1);
    lgraph = connectLayers(lgraph,startLayer, endLayer);
end

end

%--------------------------------------------------------------------------
function args = iParseInputs(varargin)

p = inputParser;
p.addRequired('imageSize', @iCheckImageSize);
p.addRequired('numClasses', @iCheckNumClasses);
p.addParameter('FilterSize', 3, @iCheckFilterSize);
p.addParameter('EncoderDepth', 4, @iCheckEncoderDepth);
p.addParameter('NumOutputChannels', 64, @iCheckNumOutputChannels);

p.parse(varargin{:});

userInput = p.Results;

imageSize = userInput.imageSize;
sizeFactor = 2^userInput.EncoderDepth;

if any(rem([imageSize(1) imageSize(2)],sizeFactor))
    error(message('vision:semanticseg:imageSizeIncompatible'));
end

args.imageSize  = double(imageSize);
args.numClasses = double(userInput.numClasses);
if isscalar(userInput.FilterSize)
    args.FilterSize = [double(userInput.FilterSize) double(userInput.FilterSize)];
end
args.EncoderDepth = double(userInput.EncoderDepth);
args.NumOutputChannels = double(userInput.NumOutputChannels);
end

%--------------------------------------------------------------------------
function iCheckImageSize(x)
validateattributes(x, {'numeric'}, ...
    {'vector', 'real', 'finite', 'integer', 'nonsparse', 'positive'}, ...
    mfilename, 'imageSize');

N = numel(x);
if ~(N == 2 || N == 3)
    error(message('vision:semanticseg:imageSizeIncorrect'));
end
end

%--------------------------------------------------------------------------
function iCheckFilterSize(x)
% require odd filter sizes to facilitate "same" output size padding. In the
% future this can be relaxed with asymmetric padding.
if isscalar(x)
    validateattributes(x, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'integer', 'nonsparse', 'positive', 'odd'}, ...
        mfilename, 'FilterSize');
else
    validateattributes(x, {'numeric'}, ...
        {'vector', 'real', 'finite', 'integer', 'nonsparse', 'positive', 'odd'}, ...
        mfilename, 'FilterSize');
end
end

%--------------------------------------------------------------------------
function iCheckNumClasses(x)
validateattributes(x, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'integer', 'nonsparse', '>', 1}, ...
    mfilename, 'numClasses');
end

%--------------------------------------------------------------------------
function iCheckNumOutputChannels(x)
validateattributes(x, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'integer', 'nonsparse', 'positive'}, ...
    mfilename, 'NumOutputChannels');
end

%--------------------------------------------------------------------------
function iCheckEncoderDepth(x)
validateattributes(x, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'integer', 'nonsparse', 'positive'}, ...
    mfilename, 'EncoderDepth');
end

%--------------------------------------------------------------------------
function [encoder, finalNumChannels] = iCreateEncoder(encoderDepth, convFilterSize, initialEncoderNumChannels, inputNumChannels)

encoder = [];
for stage = 1:encoderDepth
    % Double the layer number of channels at each stage of the encoder.
    encoderNumChannels = initialEncoderNumChannels * 2^(stage-1);
    
    if stage == 1
        firstConv = createAndInitializeConvLayer(convFilterSize, inputNumChannels, encoderNumChannels, ['Encoder-Stage-' num2str(stage) '-Conv-1']);
        % add batch normalization layers
        firstBatchNorm = batchNormalizationLayer('Name', ['Encoder-Stage-' num2str(stage) '-BatchNorm-1']);
    else
        firstConv = createAndInitializeConvLayer(convFilterSize, encoderNumChannels/2, encoderNumChannels, ['Encoder-Stage-' num2str(stage) '-Conv-1']);
        % add batch normalization layers
        firstBatchNorm = batchNormalizationLayer('Name', ['Encoder-Stage-' num2str(stage) '-BatchNorm-1']);
    end
    firstReLU = reluLayer('Name',['Encoder-Stage-' num2str(stage) '-ReLU-1']);
    
    secondConv = createAndInitializeConvLayer(convFilterSize, encoderNumChannels, encoderNumChannels, ['Encoder-Stage-' num2str(stage) '-Conv-2']);
    % add batch normalization layers
    secondBatchNorm = batchNormalizationLayer('Name', ['Encoder-Stage-' num2str(stage) '-BatchNorm-2']);
    
    secondReLU = reluLayer('Name',['Encoder-Stage-' num2str(stage) '-ReLU-2']);
    
    
    encoder = [encoder; firstConv ; firstBatchNorm; firstReLU; secondConv; secondBatchNorm ; secondReLU];
    
    if stage == encoderDepth
        dropOutLayer = dropoutLayer(0.5,'Name',['Encoder-Stage-' num2str(stage) '-DropOut']);
        encoder = [encoder; dropOutLayer];
    end
    
    maxPoolLayer = maxPooling2dLayer(2, 'Stride', 2, 'Name',['Encoder-Stage-' num2str(stage) '-MaxPool']);
    
    encoder = [encoder; maxPoolLayer];
end
finalNumChannels = encoderNumChannels;
end

%--------------------------------------------------------------------------
function [decoder, finalDecoderNumChannels] = iCreateDecoder(encoderDepth, upConvFilterSize, convFilterSize, initialDecoderNumChannels)

decoder = [];
for stage = 1:encoderDepth
    % Half the layer number of channels at each stage of the decoder.
    decoderNumChannels = initialDecoderNumChannels / 2^(stage-1);
    
    upConv = createAndInitializeUpConvLayer(upConvFilterSize, 2*decoderNumChannels, decoderNumChannels, ['Decoder-Stage-' num2str(stage) '-UpConv']);
    upReLU = reluLayer('Name',['Decoder-Stage-' num2str(stage) '-UpReLU']);
    
    % Input feature channels are concatenated with deconvolved features within the decoder.
    depthConcatLayer = depthConcatenationLayer(2, 'Name', ['Decoder-Stage-' num2str(stage) '-DepthConcatenation']);
    
    firstConv = createAndInitializeConvLayer(convFilterSize, 2*decoderNumChannels, decoderNumChannels, ['Decoder-Stage-' num2str(stage) '-Conv-1']);
    
    firstBatchNorm = batchNormalizationLayer('Name', ['Decoder-Stage-' num2str(stage) '-BatchNorm-1']);
    firstReLU = reluLayer('Name',['Decoder-Stage-' num2str(stage) '-ReLU-1']);
    
    
    secondConv = createAndInitializeConvLayer(convFilterSize, decoderNumChannels, decoderNumChannels, ['Decoder-Stage-' num2str(stage) '-Conv-2']);
    
    secondBatchNorm = batchNormalizationLayer('Name', ['Decoder-Stage-' num2str(stage) '-BatchNorm-2']);
    secondReLU = reluLayer('Name',['Decoder-Stage-' num2str(stage) '-ReLU-2']);
    
    decoder = [decoder; upConv; upReLU; depthConcatLayer; firstConv; firstBatchNorm; firstReLU; secondConv; secondBatchNorm ; secondReLU];
end
finalDecoderNumChannels = decoderNumChannels;
end

%--------------------------------------------------------------------------
function convLayer = createAndInitializeConvLayer(convFilterSize, inputNumChannels, outputNumChannels, layerName)

convLayer = convolution2dLayer(convFilterSize,outputNumChannels,...
    'Padding', 'same',...
    'BiasL2Factor',0,...
    'Name',layerName);

% He initialization is used
convLayer.Weights = sqrt(2/((convFilterSize(1)*convFilterSize(2))*inputNumChannels)) ...
    * randn(convFilterSize(1),convFilterSize(2), inputNumChannels, outputNumChannels);

convLayer.Bias = zeros(1,1,outputNumChannels);
convLayer.BiasLearnRateFactor = 2;
end

%--------------------------------------------------------------------------
function upConvLayer = createAndInitializeUpConvLayer(UpconvFilterSize, inputNumChannels, outputNumChannels, layerName)

upConvLayer = transposedConv2dLayer(UpconvFilterSize, outputNumChannels,...
    'Stride',2,...
    'BiasL2Factor',0,...
    'Name',layerName);

% The transposed conv filter size is a scalar
upConvLayer.Weights = sqrt(2/((UpconvFilterSize^2)*inputNumChannels)) ...
    * randn(UpconvFilterSize,UpconvFilterSize,outputNumChannels,inputNumChannels);
upConvLayer.Bias = zeros(1,1,outputNumChannels);
upConvLayer.BiasLearnRateFactor = 2;
end
