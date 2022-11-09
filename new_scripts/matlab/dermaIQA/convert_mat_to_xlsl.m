clear variables
close all
clc

fileToConvert = 'D:/DOMI/University/Thesis/GitHub_Repos/dermaIQA/dermaRealESRGAN_iqa.mat';
xcelFilename = 'v1_metrics.xlsx';
data = load(fileToConvert);
f = fieldnames(data);
% for k = 1:size(f,1)
%     xlswrite(xcelFilename, data.(f{k}), f{k})
% end