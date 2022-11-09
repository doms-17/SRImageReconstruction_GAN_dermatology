clear variables
close all
clc

addpath('Ma_score');
addpath('Ma_score/external/matlabPyrTools','Ma_score/external/randomforest-matlab/RF_Reg_C');
gt_path = 'D:/DOMI/University/Thesis/Inference/test_images';
sr_path = 'D:/DOMI/University/Thesis/Inference/results';

imageFormat = '*.jpg';
gtImages = read_images(gt_path, imageFormat);
srImages = read_images(sr_path, imageFormat);

metricsFR.psnr = 0;
metricsFR.ssim = 0;
metricsFR.ms_ssim = 0;
metricsFR.fsim = 0;


gt_metricsNR.niqe = 0;
gt_metricsNR.piqe = 0;
gt_metricsNR.brisque = 0;
gt_metricsNR.ma = 0;

sr_metricsNR.niqe = 0;
sr_metricsNR.piqe = 0;
sr_metricsNR.brisque = 0;
sr_metricsNR.ma = 0;

for k = 1 : length(gtImages)
    metricsFR(k).psnr = psnr(cell2mat(srImages(1,k)), cell2mat(gtImages(1,k)));
    metricsFR(k).ssim = ssim(cell2mat(srImages(1,k)), cell2mat(gtImages(1,k)));
    metricsFR(k).ms_ssim = mean(multissim(cell2mat(srImages(1,k)), cell2mat(gtImages(1,k))));
    metricsFR(k).fsim = fsim(cell2mat(srImages(1,k)), cell2mat(gtImages(1,k)));
end


for k = 1 : length(gtImages)
    gt_metricsNR(k).niqe = niqe(cell2mat(gtImages(1,k)));
    gt_metricsNR(k).piqe = piqe(cell2mat(gtImages(1,k)));
    gt_metricsNR(k).brisque = brisque(cell2mat(gtImages(1,k)));
%     gt_metricsNR(k).ma = quality_predict(cell2mat(gtImages(1,k)));
end
for k = 1 : length(srImages)
    sr_metricsNR(k).niqe = niqe(cell2mat(srImages(1,k)));
    sr_metricsNR(k).piqe = piqe(cell2mat(srImages(1,k)));
    sr_metricsNR(k).brisque = brisque(cell2mat(srImages(1,k)));
%     sr_metricsNR(k).ma = quality_predict(cell2mat(srImages(1,k)));
end


fprintf(1, '\nFinished\n');


