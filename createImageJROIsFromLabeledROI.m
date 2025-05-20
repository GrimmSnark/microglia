function createImageJROIsFromLabeledROI(labeledROI,RM, clearROIs, downsampleVertNum)
% Generates a new set of ROIs in ImageJ's ROI Manager from the input
% labeledROI.
%
% Example: createImageJROIsFromLabeledROI(labeledROI,RC)
% 
% by David Whitney (david.whitney@mpfi.org), Max Planck Florida Institute, 2016.

if nargin < 3 || isempty(clearROIs)
    clearROIs = 1;
end

if nargin < 4 || isempty(downsampleVertNum)
    downsampleVertNum = NaN;
end


newROIs=[];
uniqueValues=unique(labeledROI(labeledROI>0));
for(i=1:length(uniqueValues))
    boundaryPts = bwboundaries(labeledROI==uniqueValues(i));


    % downsample if required
    if ~isnan(downsampleVertNum)
        if length(boundaryPts{:})>downsampleVertNum 
            lenVerts = length(boundaryPts{:});
            target = downsampleVertNum;

            sampIndexes = round(interp1( 1:lenVerts, linspace(1, lenVerts, target)));

            boundaryPts{1} = boundaryPts{1}(sampIndexes,:);
        end
    end

    boundaryPts = boundaryPts{1}-1; % subtract one because Java arrays start from 0, rather than 1.
    newROIs = cat(1,newROIs,ij.gui.PolygonRoi(boundaryPts(:,2),boundaryPts(:,1),size(boundaryPts,1),ij.gui.Roi.POLYGON));
end

% Clear existing ROIs
if clearROIs == 1
    if RM.getCount~=0
        RM.runCommand('Deselect');
        RM.runCommand('Delete');
    end
end

% Add new ROIs
for i=1:length(newROIs)
    RM.addRoi(newROIs(i));
end

end