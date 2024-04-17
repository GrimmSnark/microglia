function cleanMicrogliaCSV_v2(csvFilepath)
% This function cleans the csv file created by the microglia morphometry
% plugin. The file has to have been run through the cluster cell/blood
% vessel area app first. The script sets a bunch of remove flags for
% objects which are too small and appear for too few frames (see defaults)
%
% Inputs: csvFilepath- fullfile to the microglia morphometry .csv file OR
%                      leave blank to open a file selection window
%
% Written by MA Savage
%% defaults
if nargin < 1 || isempty(csvFilepath)
   [file, path] = uigetfile({'*.csv'},...
                          'Image File Selector');

   csvFilepath = fullfile(path,file);
end

areaLim = 100; % pixels ^2
frameNo = 3;
imageSz = 2048; % in pixels
boundaryLim = 25;   % in microns
FrameIntervalDefault = 60; % default of 60s if cannot find it from the image metadata

%% read in data
microgliaTab = readtable(csvFilepath);

csvHeight = height(microgliaTab);
microgliaTab.object2Remove = zeros(csvHeight,1);
%% clean data

% remove objects by size
microgliaTab.object2Remove(microgliaTab.Area_Pixel2 <= areaLim) = 1;

% remove objects by number of frames
objLab = unique(microgliaTab.Object_Label);

% get eucildian image boundary
boundaryX = [ones(1,imageSz) 1:imageSz (ones(1,imageSz)* imageSz) imageSz:-1:1];
boundaryY = [1:imageSz (ones(1,imageSz)* imageSz) imageSz:-1:1 ones(1,imageSz)];

% pixel limit conversion
pixelLim = boundaryLim/microgliaTab.VoxelSpacing_X(1);

% for each object
for i = 1:length(objLab)
    obj2Check = microgliaTab(microgliaTab.Object_Label == objLab(i),:);

    % check if it is in the recording for longer than frameNo
    if height(obj2Check) < frameNo
        microgliaTab.object2Remove(microgliaTab.Object_Label == objLab(i)) = 1;
    end

    % get the minmum boundary distance for each frame object
    distPerFrame = pdist2([boundaryX;boundaryY]',[obj2Check.Centroid_X_Pixel obj2Check.Centroid_Y_Pixel],'euclidean', 'Smallest',1);

    if sum(distPerFrame < pixelLim) > 1
        microgliaTab.object2Remove(microgliaTab.Object_Label == objLab(i)) = 1;
    end
end

%% calculate other metrics

% add new table columns
microgliaTab.Area_Micron2 = zeros(csvHeight,1);
microgliaTab.ConvexArea_Micron2 = zeros(csvHeight,1);
microgliaTab.GeodesicDiameter_Micron = zeros(csvHeight,1);
microgliaTab.Perimeter_Micron = zeros(csvHeight,1);
microgliaTab.SkeletonAvgBranchLength_Micron = zeros(csvHeight,1);
microgliaTab.SkeletonLongestBranchLength_Micron = zeros(csvHeight,1);
microgliaTab.SkeletonTotalLength_Micron = zeros(csvHeight,1);
microgliaTab.DistanceMovePix = zeros(csvHeight,1);
microgliaTab.DistanceMoveMicron =  zeros(csvHeight,1);
microgliaTab.velocityPerFrameMicronPerSec = zeros(csvHeight,1);
microgliaTab.circularity = zeros(csvHeight,1);
microgliaTab.somaness = zeros(csvHeight,1);
microgliaTab.branchiness = zeros(csvHeight,1);


for i = 1:length(objLab)
    tempTab = microgliaTab(microgliaTab.Object_Label == objLab(i),:);

    P1 = [tempTab.Centroid_X_Pixel tempTab.Centroid_Y_Pixel];
    % add other metrics

    tempTab.Area_Micron2 = tempTab.Area_Pixel2 * tempTab.VoxelSpacing_X(1);
    tempTab.ConvexArea_Micron2 = tempTab.ConvexArea_Pixel2 * tempTab.VoxelSpacing_X(1);

    try
        tempTab.GeodesicDiameter_Micron = tempTab.GeodesicDiameter_Pixel * tempTab.VoxelSpacing_X(1);
    catch

    end
    tempTab.Perimeter_Micron = tempTab.Perimeter_Pixel * tempTab.VoxelSpacing_X(1);
    tempTab.SkeletonAvgBranchLength_Micron = tempTab.SkeletonAvgBranchLength_Pixel * tempTab.VoxelSpacing_X(1);
    tempTab.SkeletonLongestBranchLength_Micron = tempTab.SkeletonLongestBranchLength_Pixel * tempTab.VoxelSpacing_X(1);
    tempTab.SkeletonTotalLength_Micron = tempTab.SkeletonTotalLength_Pixel * tempTab.VoxelSpacing_X(1);


    % distance moved in pixels
    dists = pdist2(P1, P1,"euclidean" );
    inxd1 = 2:length(dists);
    indx2 = 1:length(dists)-1;
    indFromSub = sub2ind(size(dists),inxd1,  indx2);

    tempTab.DistanceMovePix = [0 dists(indFromSub)]';

    tempTab.DistanceMoveMicron =  tempTab.DistanceMovePix * tempTab.VoxelSpacing_X(1);

    % checks if the frame interval was imported properly from the image
    % metadata
    if tempTab.FrameInterval(1) == 0 || isempty(tempTab.FrameInterval(1))
        tempTab.velocityPerFrameMicronPerSec = tempTab.DistanceMoveMicron ./FrameIntervalDefault;
    else
        tempTab.velocityPerFrameMicronPerSec = tempTab.DistanceMoveMicron ./tempTab.FrameInterval;
    end

    tempTab.circularity = (4 * pi * tempTab.Area_Pixel2) ./(tempTab.Perimeter_Pixel .^2);

    tempTab.somaness = (tempTab.RadiusAtBrightestPoint_Pixel .^2) ./tempTab.Area_Pixel2;

    try
        tempTab.branchiness = tempTab.SkeletonNumBranchPoints ./ tempTab.GeodesicDiameter_Pixel;
    catch
    end

    microgliaTab(microgliaTab.Object_Label == objLab(i),:) = tempTab;
end

%% clean to only valid objects
microgliaTabValid = microgliaTab(microgliaTab.object2Remove == 0,:);

%% save the struct 

[folder, name] = fileparts(csvFilepath);

if endsWith(name,'_cleaned')
    writetable(microgliaTab,fullfile(folder,[name '.xlsx']));
     writetable(microgliaTabValid,fullfile(folder,[name '_valid.xlsx']));
else
    writetable(microgliaTab,fullfile(folder,[name '_cleaned.xlsx']));
    writetable(microgliaTabValid,fullfile(folder,[name '_cleaned_valid.xlsx']));
end

end