function countYoPro_HMOX1(HMOX1ImPath, yoproImPath)

%% defaults
sizeLim = 50;
props = ["Area", "Centroid", "PixelIdxList"];
[savePath, name] = fileparts(HMOX1ImPath);

%% read in 
yoproIm = read_Tiffs(yoproImPath);
HMOX_Im = read_Tiffs(HMOX1ImPath);

%% get objects
yoproRP = regionprops(yoproIm, props);
HMOX_RP = regionprops(HMOX_Im, props);

if length(yoproRP) == 255 || length(yoproRP) == 2
    yoproCC = bwconncomp(yoproIm);
    yoproRP = regionprops(yoproCC, props);
end

if length(HMOX_RP) == 255 || length(HMOX_RP) == 2
    HMOX_CC = bwconncomp(HMOX_Im);
    HMOX_RP = regionprops(HMOX_CC, props);
end

%% Filter for size
yoproRP([yoproRP.Area] < sizeLim) = [];
HMOX_RP([HMOX_RP.Area] < sizeLim) = [];

%% check overlaps
overlaps = zeros(length(HMOX_RP),1);
for i = 1:length(HMOX_RP)
    count = 1;
    for j = 1:length(yoproRP)
        tempHO = HMOX_RP(i).PixelIdxList;
        tempYoPro = yoproRP(j).PixelIdxList;

        tempOverlaps = length(tempHO)/sum(ismember(tempYoPro,tempHO));

        if ~isinf(tempOverlaps)
            overlaps(i, count) = tempOverlaps;
            count = count+1;
        end
    end
end

overlapCells = any(max(overlaps,[],2),2);
%% create counts

numOfHO = length(HMOX_RP);
numOfYoPro = length(yoproRP);
numOverlap= sum(overlapCells);


table2Save = table(numOfHO,numOfYoPro,numOverlap);

% save
saveName = regexp(name, '^(.*)_C1', 'tokens', 'once');

writetable(table2Save, fullfile(savePath, [saveName{1} '_cellNumbers.xlsx']))

end