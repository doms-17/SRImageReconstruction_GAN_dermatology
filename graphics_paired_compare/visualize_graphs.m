clc
clear variables
close all

path = "D:\DOMI\University\Magistrale\Tesi\Pipeline_coding\dataset_paired_sliding\";
folder = "\lowResolution_x8\";
NUM_IMGS = 3;

filesList = dir(fullfile(path+folder, '*.png'));
idx = randperm(numel(filesList), NUM_IMGS);

imageLists = [""];

for k = 1:NUM_IMGS
  complete_path = fullfile(path, filesList(idx(k)).name);
  id_file = split(complete_path,"\")';
  imageLists(k) = id_file(1,end);
end

finalArray = [];
for k = 1:NUM_IMGS
    folder = "\lowResolution_x8\";
    lowres_8 = im2double(imread(path+folder+imageLists(k)));
    folder = "\lowResolution_x4\";
    lowres_4 = im2double(imread(path+folder+imageLists(k)));
    folder = "\highResolution\";
    highres = im2double(imread(path+folder+imageLists(k)));
    folder = "\highResolution_plus\";
    highres_plus = im2double(imread(path+folder+imageLists(k)));
    imagesArray = [lowres_8,lowres_4,highres,highres_plus];
    finalArray = cat(1,imagesArray, finalArray);
end

figure
montage({finalArray},'Size', [1 NaN]);
img_type_name = ["lowResx8", "lowresx4", "highRes", "highRes+"];
imgs_name = [""];
for i = 1:numel(imageLists)
    new_str = split(imageLists(i),".")';
    new_str = split(new_str(1),"_")';
    imgs_name(i) = new_str(1)+new_str(2);
%     imgs_name(i) = strrep(new_str(1),"_","");
end

offset=0;
for i=1:numel(imgs_name)
    text(-120,100+offset,imgs_name(i),'fontsize',10,'color','black','fontweight','bold') %works perfectly
    offset=offset+300;
end
offset=0;
for i=1:numel(img_type_name)
    text(70+offset,-15,img_type_name(i),'fontsize',10,'color','black','fontweight','bold') %works perfectly
    offset=offset+300;
end

saveas(gcf,'Images_compare.png')