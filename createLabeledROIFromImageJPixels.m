function [labeledROI, centerXY] = createLabeledROIFromImageJPixels(imgSize,roiObjects)
% Creates a labeled ROI from the ROIs defined in ImageJ's ROI Manager.
%
% Example: labeledROI = createLabeledROIFromImageJPixels([512 512],RC.getRoisAsArray)
%
% by David Whitney (david.whitney@mpfi.org), Max Planck Florida Institute, 2016.

labeledROI = zeros(imgSize);
nROIs = length(roiObjects);
for(i=1:nROIs)
    % Get center location for ROI object
    X = round(roiObjects(i).getXBase+1); % add one because MATLAB arrays start at 1, while Java arrays start at 0.
    Y = round(roiObjects(i).getYBase+1);
    
    % Get local mask for ROI object
    try
    localCellMask = roiObjects(i).getMask();
        height = localCellMask.getHeight();
    width  = localCellMask.getWidth();
    boundedPixels = double(localCellMask.getPixels());
    localCellImg = reshape(boundedPixels,[width,height]);
    localCellImg(localCellImg==-1) = i;
    
    % HACK HACK
    if X<0
        X = 0;
    end
    
    if Y < 0
        Y = 0;
    end
    
    %     labeledROI(Y+[1:height],X+[1:width]) = localCellImg';
    
    % Only adds elements that are non zero, stops bounding box overlap
    localCellImg = localCellImg';
    for x=1:numel(localCellImg)
        if localCellImg(x) ~=0
            [currentW, currentH] = ind2sub([ height width],x);
            labeledROI(Y+currentW,X+currentH) = localCellImg(x);
        end
    end
    
    labeledROI = labeledROI(1:imgSize(1), 1:imgSize(2));
    
    centerXY(i,:) = [X Y];
    catch
        
    end
    %     imagesc(labeledROI);
end