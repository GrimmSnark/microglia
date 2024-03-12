function reclassedIm = trackMicrogliaMasksSingleIm(maskPath, erodeFlag)

% defaults
if nargin < 2
    erodeFlag = 0;
end

hardCentroidDistLimPix = 75; % search radius in pixels

try
    gpuArray(1);
    canUseGPU=true;
catch
    canUseGPU=false;
end

%% load in data

if isstring(maskPath)
    masks = read_Tiffs(maskPath);

    if canUseGPU == true
        masks = gpuArray(masks);
    end
else
    if canUseGPU == true
        masks = gpuArray(maskPath);
    else
        masks = maskPath;
    end
end

%% make it binary

masks = imcomplement(logical(masks)); % Threshold.

%% get improps

if erodeFlag == 1
    % erode image by 1 pixel
    se = strel('disk', 1);

    masks = imerode(masks, se);
end

if canUseGPU == 1
    tempImProps =  bwconncomp(gather(masks), 4);
else
    tempImProps =  bwconncomp(masks, 4);
end

% rebuild the now split image
currImage = labelmatrix(tempImProps);

% clean image props to remove blanks and very small objects
currImLens = cellfun(@length, tempImProps.PixelIdxList);
currImFilterNums = find(currImLens < 100);

% replace all small numbers with zeros
[replaceIndx, replaceLocs] = ismember(currImage, currImFilterNums);
replaceNumbers =  [zeros(length(currImFilterNums),1)];
currImage(replaceIndx) = replaceNumbers(replaceLocs(replaceIndx));

% reindex the objects
[replaceIndx, replaceLocs] = ismember(currImage, unique(currImage));
replaceNumbers =  [0:length(unique(currImage))-1];
currImage(replaceIndx) = replaceNumbers(replaceLocs(replaceIndx));

reclassedIm = currImage;
end