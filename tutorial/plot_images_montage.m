clc
clear variables
close all

addpath('Branciforti Scripts\')

% main path
path = "D:\DOMI\University\Thesis\Presentation_Thesis\Images_graphs_tables\Materials and methods";
folder = "\data_analysis\";     % subfolder
NUM_IMGS = 12;   % number of images to plot

filesList = dir(fullfile(path+folder, '*.png'));
idx = randperm(numel(filesList), NUM_IMGS);

imageLists = [""];

for k = 1:NUM_IMGS
  complete_path = fullfile(path, filesList(idx(k)).name);
  id_file = split(complete_path,"\")';
  imageLists(k) = id_file(1,end);
end

imagesArray = {};
for k = 1:NUM_IMGS
    img = im2double(imread(path+folder+imageLists(k)));
    imagesArray{k} = img;
end

figure
montage(imagesArray,'Size', [3 NaN]);

% saveas(gcf,'Images_compare.png')