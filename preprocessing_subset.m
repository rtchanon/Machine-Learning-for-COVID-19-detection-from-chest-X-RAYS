
Dest     = 'COVID-19_Radiography_Dataset\new';
FileList = dir(fullfile('COVID-19_Radiography_Dataset\Normal', '*.png'));
Index    = randperm(numel(FileList), 3616);
for k = 1:3616
  Source = fullfile('COVID-19_Radiography_Dataset\Normal', FileList(Index(k)).name);
  copyfile(Source, Dest);
end