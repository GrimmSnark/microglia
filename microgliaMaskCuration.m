function microgliaMaskCuration(labelMasksPath, invertIm)

%% defaults
if nargin < 1 || isempty(labelMasksPath)
    [file, path] = uigetfile({'*.tif'},...
        'Label Masks File Selector');

    labelMasksPath = fullfile(path,file);
end

if nargin < 2  || isempty(invertIm)
    invertIm = 0;
end

intializeMIJ;

%% load in data

[fileFolder, fileName] = fileparts(labelMasksPath);
timelapsePath = fullfile(fileFolder, [fileName(1:end-11) '.tif']);

masksTif = read_Tiffs(labelMasksPath);
timelapseTif = read_Tiffs(timelapsePath);


%% run first classification
tic
reclassImage = trackMicrogliaMasks(masksTif,[],invertIm);
toc

%% create LUT image
% IDs = unique(reclassImage(:));
% IDs = [IDs' IDs(end)+1: IDs(end)+100];
% testIm = uint16(zeros(size(reclassImage,[1 2])));
% 
% LUT = distinguishable_colors(max(IDs),'k');
% reclassImage = cat(3,testIm,reclassImage);
%% load into FIJI
reclassImageImp = MIJ.createImage('Label Masks',reclassImage,1);
% reclassImageImp.setLut( LUT)
ij.IJ.run("Glasbey on Dark");
ij.IJ.run("Macro...", "code=v=v%255");

% ij.IJ.run(reclassImageImp, "Glasbey modulus for 16bit count masks", "");

timelapseImageImp = MIJ.createImage('Timelapse',timelapseTif,1);
ij.IJ.run("Synchronize Windows", "");

%% while loop to keep doing manual curation/classification
happy = false;

while ~happy

      % Query user if you want to use previously chosen ROIs
    answer = MFquestdlg([0.5,0.5], 'How would you like to proceed?', ...
        'Use the Microglia Mask Modifier Plugin to make the masks better then proceed', ...
        'Recalculate Masks','Save Masks', 'Recalculate Masks');
    % Handle response
    switch answer
        case 'Recalculate Masks'
            modifiedStack = MIJ.getImage('Label Masks');
            reclassImage = trackMicrogliaMasks(modifiedStack);

            reclassImageImp.changes = false;
            reclassImageImp.close;
            reclassImageImp = MIJ.createImage('Label Masks',reclassImage,1);

            ij.IJ.run("Glasbey on Dark");
            ij.IJ.run("Macro...", "code=v=v%255");

%             ij.IJ.run(reclassImageImp, "Glasbey modulus for 16bit count masks", "");
            
        case 'Save Masks and Close'
            happy = true;

        case 'Save and Continue Curation'
            saveIm = MIJ.getImage('Label Masks');
            saveastiff(saveIm, labelMasksPath);

        case ''

    end
end

% add kill button


%% save the masks
saveIm = MIJ.getImage('Label Masks');
saveastiff(saveIm, labelMasksPath);

reclassImageImp.changes = false;
reclassImageImp.close;

timelapseImageImp.changes = false;
timelapseImageImp.close;

% %% run metrics morphology scripts
% ij.IJ.run('Measure Microglia Morphometry', ['intensityfile= "' '\\campus\rdw\ion10\10\retina\data\microglia\Cori microglia analysis\Copy folder\Substack (1-5)_intensity.tif' '" labelmaskfile= "' labelMasksPath '" outputdirectory= "' fileFolder '"']);
% 
% ij.IJ.run('Measure Microglia Morphometry', ['intensityfile= "' '\\campus\rdw\ion10\10\retina\data\microglia\Cori microglia analysis\Copy folder\Substack (1-5)_intensity.tif' '" labelmaskfile= "' '\\campus\rdw\ion10\10\retina\data\microglia\Cori microglia analysis\Copy folder\Substack (1-5).tif' '" outputdirectory= "' fileFolder '"']);


end
