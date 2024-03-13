function reclassImage = trackMicrogliaMasks(maskPath, erodeFlag, invertIm)

% defaults
if nargin < 2
    erodeFlag = 0;
end

if nargin < 3  || isempty(invertIm)
    invertIm = 0;
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

for fr = 1:size(masks,3)
    if invertIm == 1
        binaryImage = imcomplement(logical(masks(:,:,fr))); % Threshold.
    else
        binaryImage = logical(masks(:,:,fr)); % Threshold.
    end
    masks(:,:,fr) = binaryImage;
end

%% get improps

if erodeFlag == 1
    % erode image by 1 pixel
    se = strel('disk', 1);

    for fr = 1:size(masks,3)
        mask = imerode(binaryImage, se);
        masks(:,:,fr) = mask;
    end
end


% for each image
for i = 1: size(masks,3)-1

    disp(['On Frame ' num2str(i) ' of ' num2str(size(masks,3))]);
    % if we are looking at first image, use the raw masks
    if i == 1
        currImage = masks(:,:,i);
        currImage = logical(currImage);


        if canUseGPU == 1
            tempImProps =  bwconncomp(gather(currImage), 4);
        else
            tempImProps =  bwconncomp(currImage, 4);
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

        reIndexStack = currImage;
    else % otherwise use the last image in the cleaned aligned stack

        currImage = reIndexStack(:,:,i);
    end


    nextImage = masks(:,:,i+1);
    nextImage = logical(nextImage);

    if canUseGPU == 1
        nextImProps =  bwconncomp(gather(nextImage), 4);
    else
        nextImProps =  bwconncomp(nextImage, 4);
    end

     % rebuild the now split image
    nextImage = labelmatrix(nextImProps);

    % clean image props to remove blanks and very small objects
    nextImLens = cellfun(@length, nextImProps.PixelIdxList);
    nextImFilterNums = find(nextImLens < 100);


    % replace all small numbers with zeros
    [replaceIndx, replaceLocs] = ismember(nextImage, nextImFilterNums);
    replaceNumbers =  [zeros(length(nextImFilterNums),1)];
    nextImage(replaceIndx) = replaceNumbers(replaceLocs(replaceIndx));

    % reindex the objects
    [replaceIndx, replaceLocs] = ismember(nextImage, unique(nextImage));
    replaceNumbers =  [0:length(unique(nextImage))-1];
    nextImage(replaceIndx) = replaceNumbers(replaceLocs(replaceIndx));

    %     % get filtered image objects
    %     curImPropsFiltered = struct2cell(regionprops(currImage, "PixelIdxList"));
    %     nextImPropsFiltered = struct2cell(regionprops(nextImage, "PixelIdxList"));

    % run through all objects in current and next frame looking for
    % highest correlation matches

    currentIndexes = unique(currImage);
    currentIndexes = currentIndexes(2:end); % remove the 0 from the index list

    nextIndexes = unique(nextImage);
    nextIndexes = nextIndexes(2:end); % remove the 0 from the index list

    corrMat = [];

    % get the center distances to use a filter in whether to try
    % correlation
    currImCenter = regionprops(currImage,"Centroid");
    currImCenter = [currImCenter.Centroid];
    currImCenter = reshape(currImCenter,2,[])';

    nextImCenter = regionprops(nextImage,"Centroid");
    nextImCenter = [nextImCenter.Centroid];
    nextImCenter = reshape(nextImCenter,2,[])';

    distCenters = pdist2(currImCenter, nextImCenter, "euclidean" );
    distCenters = distCenters(currentIndexes,:);

    nIndLen = length(nextIndexes);
    parfor ob = 1:length(currentIndexes) % clean this up here so remove if statement....
        %     for ob = 1:length(currentIndexes) % clean this up here so remove if statement....

        for n = 1:nIndLen

            if distCenters(ob,n) < hardCentroidDistLimPix
                curImOb = ismember(currImage, currentIndexes(ob));
                nextImOb = ismember(nextImage,  nextIndexes(n));

                corrMat(ob, n) = corr2(curImOb, nextImOb);

            end
        end
    end

    [maxCorr, matchIndx] = max(corrMat, [],2);

    % build the next aligned stack for the current frames

    if canUseGPU == true
        maxInd = double(gather(max(reIndexStack(:)))+1); % has to be double for multiplication below
    else
        maxInd = max(reIndexStack(:))+1;
    end

    %     maxInd = max(reIndexStack(:))+1;

    currIndx = []; % blank out before filling for each frame

    for ee = 1:length(nextIndexes)  % for all objects in the next image
        if ismember(nextIndexes(ee),matchIndx) % see if it has a match in the previous image

            % get the matching index for this object
            currentMatchingIndx = currentIndexes(find(matchIndx == nextIndexes(ee)));

            % if there are more than two choose the one that has the
            % highest correlation
            if length(currentMatchingIndx) > 1
                overlappedIndx = find(matchIndx == nextIndexes(ee)); % indexing within the unique numbers in the next image
                [~, tempIndx] = max(maxCorr(overlappedIndx));
                currentMatchingIndx = currentIndexes(overlappedIndx(tempIndx));
            end

            if canUseGPU == true
                currentMatchingIndx = gather(currentMatchingIndx);  % has to be double for multiplication below
            end

            currIndx(:,:,ee) = ismember(nextImage, nextIndexes(ee)) * double(currentMatchingIndx);

            %             nextImOb = gather(ismember(nextImage,  ee));
            %             prevImOb = gather(ismember(currImage, currentIndexes(find(matchIndx == ee))));
            %             C = imfuse(prevImOb,nextImOb,'falsecolor','Scaling','joint','ColorChannels',[1 2 0]);
            %             imshow(C);
            %             C;
        else
            currIndx(:,:,ee) = ismember(nextImage, ee)* double(maxInd) ;
            maxInd = maxInd +1;
        end
    end

    % get the max for the next frame
    reIndexStack(:,:,i+1) = max(currIndx,[],3);
end

if canUseGPU == true
    reclassImage = gather(reIndexStack);
else
    reclassImage = reIndexStack;
end

% saveastiff(saveIm, '\\campus\rdw\ion10\10\retina\data\microglia\Cori microglia analysis - Copy\20230220\Control1\testAlign.tif');
end