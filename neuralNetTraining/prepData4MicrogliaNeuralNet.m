function prepData4MicrogliaNeuralNet(labelFilepathsStruct, saveDir)


% Get all data in
all = [];
for q = 1:length(labelFilepathsStruct)
    labelStacks{q} = read_Tiffs(labelFilepathsStruct{q});
    microgliaStacks{q} = read_Tiffs([labelFilepathsStruct{q}(1:end-15) '.tif']);
end

count = 1;
for q = 1:length(labelFilepathsStruct)
    for i = 1:size(labelStacks{q},3)
        labelTif = labelStacks{q}(:,:,i);
        microgliaTif  = microgliaStacks{q}(:,:,i);

        labelTif = uint8(imbinarize(labelTif) * 2^8);
        microgliaTif = imadjust(microgliaTif);

        saveastiff(labelTif, [saveDir '\masks\mask_' sprintf('%03d',count) '.tif']);
        saveastiff(microgliaTif, [saveDir '\images\image_' sprintf('%03d',count) '.tif']);


        count = count+1; 
    end
end

end