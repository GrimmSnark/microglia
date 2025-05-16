function calculateYoPro_D1_2_3_microglia(folderPath)
% Calculates the D1/D2 ratio for yopro-1 stained retinal
% wholemounts. This will run on a full folder of images. Requires each 
% image in the folder to be processed for to a binary image with cells in 
% black

%% open FIJI
% initalize MIJI
intializeMIJ;

RM = ij.plugin.frame.RoiManager();
RC = RM.getInstance();
RC.reset();

%% find the images we want to use ie 'mask'
maskFilepath = dir([folderPath '\*YP-label*']);

if isempty(maskFilepath)
    maskFilepath = dir([folderPath '\*YP-label*']);
end

% maskFilepath = dir([folderPath '**\*mask*']);
% 
% if isempty(maskFilepath)
%     maskFilepath = dir([folderPath '**\*Mask*']);
% end
% 

for x = 1:length(maskFilepath)
    filePathTemp = maskFilepath(x);

    pattern = '[pP](\d+)';
    pDay = regexp(filePathTemp.name, pattern, 'tokens');
    pDay = str2double(pDay{1}{:});

    filePathTemp = fullfile(filePathTemp.folder, filePathTemp.name);

    masks = read_Tiffs(filePathTemp);
    % masks = ~masks;
    yoproCells = bwconncomp(masks);
    yoproLabelled = labelmatrix(yoproCells);
    pixelImSize = yoproCells.ImageSize;
    centroidStruct = regionprops(yoproLabelled,"Centroid");
    maskCentroid = vertcat(centroidStruct.Centroid);

    %% Get the optic nerve head and retinal bounds
    % RC.open([filePathTemp(1:end-9) '_ROIs.zip']);
    roiFilepath = dir([folderPath '\*.zip']);
    RC.open(fullfile(roiFilepath.folder,roiFilepath.name));

    ROIobjects = RC.getRoisAsArray;

    % optic nerve poly
    opticNerveMask = createLabeledROIFromImageJPixels(yoproCells.ImageSize ,ROIobjects(1));
    centroidStruct = regionprops(opticNerveMask,"Centroid");
    opticNerveCentroid = centroidStruct.Centroid;

     % blood vessel boundary poly
    bloodVesselBoundMask = createLabeledROIFromImageJPixels(yoproCells.ImageSize  ,ROIobjects(2));
    bloodVesselBoundPoly = bwboundaries(bloodVesselBoundMask');
    bloodVesselBoundShape = polyshape(bloodVesselBoundPoly{1});

    % retina boundary poly
    retinaBoundMask = createLabeledROIFromImageJPixels(yoproCells.ImageSize  ,ROIobjects(3));
    retinaBoundPoly = bwboundaries(retinaBoundMask');
    retinaBoundShape = polyshape(retinaBoundPoly{1});


    %% 

    distFromCenter = pdist2(opticNerveCentroid,maskCentroid);
    count = 1;
    center2RetinaEdge = [];
    retinaEdgePos = [];
    pFit = [];
    for w = 1:length(maskCentroid)

        pFit(w,:,:) = fitStraightLine(opticNerveCentroid, maskCentroid(w,:), [0 max(pixelImSize)]);

        % get the points in and out of the retina shape
        currLine = squeeze(pFit(w,:,:));
        [inR, outR] = intersect(retinaBoundShape,  currLine);

        [inBV, outBV] = intersect(bloodVesselBoundShape,  currLine);

        % plot(retinaBoundShape)
        % hold on
        % scatter(maskCentroid(w,1), maskCentroid(w,2));
        % scatter(opticNerveCentroid(1), opticNerveCentroid(2));
        % plot(currLine(:,1), currLine(:,2));
        % close;

        % error catch for objects outside of retina bounds
        if isempty(inR)
            continue
        end
        % %%%%

        % D2
        [retBoundDist,indUsed] = pdist2(inR, maskCentroid(w,:),'euclidean','Smallest',1);
        line2UseD2 = [opticNerveCentroid; inR(indUsed,:) ];

        center2RetinaEdge(count) = pdist2(inR(indUsed,:), opticNerveCentroid);
        retinaEdgePos(count,:) = inR(indUsed,:);


        %D3
        [BVBoundDist,indUsed] = pdist2(inBV, maskCentroid(w,:),'euclidean','Smallest',1);
        line2UseD3 = [opticNerveCentroid; inBV(indUsed,:) ];

        center2BVEdge(count) = pdist2(inBV(indUsed,:), opticNerveCentroid);
        BVEdgePos(count,:) = inBV(indUsed,:);


        % use this to check calculations

        % g = imshow(imbinarize(masks));
        % hold on
        % plot(retinaBoundShape)
        % plot(bloodVesselBoundShape)
        % scatter(opticNerveCentroid(1), opticNerveCentroid(2));
        % scatter(maskCentroid(w,1), maskCentroid(w,2));
        % scatter(BVEdgePos(w,1),BVEdgePos(w,2));
        % scatter(retinaEdgePos(w,1),retinaEdgePos(w,2),'g')
        % 
        % plot(line2UseD2(:,1),line2UseD2(:,2),'b');
        % plot(line2UseD3(:,1),line2UseD3(:,2),'g');

        count = count +1;

    end

    distFromCenter = distFromCenter';
    center2RetinaEdge =center2RetinaEdge';
    center2BVEdge = center2BVEdge';

    %% build table
    relDistanceTab = table;
    relDistanceTab = table(maskCentroid, distFromCenter, pFit, center2RetinaEdge , retinaEdgePos,center2BVEdge, BVEdgePos);
    relDistanceTab.pDay = repmat(pDay,height(relDistanceTab),1);
    relDistanceTab.D1_2 = relDistanceTab.distFromCenter./relDistanceTab.center2RetinaEdge;
    relDistanceTab.D1_3 = relDistanceTab.distFromCenter./relDistanceTab.center2BVEdge;

    % clean numbers
    relDistanceTab(relDistanceTab.D1_2>1,:) = [];
    relDistanceTab(relDistanceTab.D1_2<0,:) = [];
    relDistanceTab(relDistanceTab.D1_3<0,:) = [];

    D1_2Table = table();
    D1_2Table.pDay = relDistanceTab.pDay;
    D1_2Table.D1_2 = relDistanceTab.D1_2;
    D1_2Table.D1_3 = relDistanceTab.D1_3;

    %% save as excel

    writetable(D1_2Table, [filePathTemp(1:end-9) '_D1_2_v2.xlsx']);

    RC.close
end
end