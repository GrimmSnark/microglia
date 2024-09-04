function filterGFP_HMXO_MicrogliaCSV()
% Checks the overlap of GFP positive microglia and HMOX-1 positive and
% removes HMOX-1 postive ones from the GFP csv 


%% defaults

areaLim = 100; % pixels ^2 
distanceLim = 100 ;% at 0.3 pix/micron = 30 micron

%% load in csv
[fileGFP, pathGFP] = uigetfile({'*.csv'},...
    'GFP File Selector');

GFP_Filepath = fullfile(pathGFP,fileGFP);

[fileHM, pathHM] = uigetfile({'*.csv'},...
    'HMOX File Selector');

HM_Filepath = fullfile(pathHM,fileHM);


GFP_Table = readtable(GFP_Filepath);
HM_Table = readtable(HM_Filepath);

%% filter
GFP_Table(GFP_Table.Area_Pixel2 <= areaLim,:) = [];
HM_Table(HM_Table.Area_Pixel2 <= areaLim,:) = [];


[D,I] = pdist2([HM_Table.Centroid_X_Pixel HM_Table.Centroid_Y_Pixel],[GFP_Table.Centroid_X_Pixel GFP_Table.Centroid_Y_Pixel],'euclidean','Smallest',1);
D = D';

overlapFlag = zeros(height(GFP_Table),1);
overlapFlag(D<distanceLim) = 1;

%% plotting (USE FOR DEBUG)

% scatter(GFP_Table.Centroid_X_Pixel(overlapFlag == 0),GFP_Table.Centroid_Y_Pixel(overlapFlag == 0),"+" );
% hold on
% scatter(HM_Table.Centroid_X_Pixel,HM_Table.Centroid_Y_Pixel, "o" );
% axis equal
% set(gca, 'YDir','reverse');
% 
% figure
% scatter(GFP_Table.Centroid_X_Pixel,GFP_Table.Centroid_Y_Pixel,"+" );
% hold on
% scatter(HM_Table.Centroid_X_Pixel,HM_Table.Centroid_Y_Pixel, "o" );
% axis equal
% set(gca, 'YDir','reverse');


%% saving filtered data

GFP_Table(overlapFlag ==1,:)=[];

writetable(GFP_Table, [GFP_Filepath(1:end-4) '_filtered.csv']);

end