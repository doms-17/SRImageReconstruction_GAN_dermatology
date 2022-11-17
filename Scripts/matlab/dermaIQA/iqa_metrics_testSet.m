clear variables
close all
clc

tic

%% ========== Utils ==========%
addpath('Ma_score');
addpath('Ma_score/external/matlabPyrTools','Ma_score/external/randomforest-matlab/RF_Reg_C');

gt_root = "D:/DOMI/University/Thesis/Coding/Dataset/TestSet/";
sr_root = "D:/DOMI/University/Thesis/Coding/Dataset/Inference/TestSet/";
testSet_name = ["Atlas","Novara_good","Novara_trash","PH2","Nurugo"];
gt_path = gt_root + testSet_name;
sr_path = sr_root + testSet_name;

scale_gt = 1;
scale_sr = 1;

%% ========== Read images ==========%
imageFormat = ["*.jpg"; "*.png"];
gtImages = {};
srImages = {};
for f = 1:size(imageFormat,1)
    gtImages_tmp = read_images(gt_path, imageFormat(f,:)); %, scale_gt);
    srImages_tmp = read_images(sr_path, imageFormat(f,:)); %, scale_sr);
    gtImages = [gtImages, gtImages_tmp];
    srImages = [srImages, srImages_tmp];
end

%% ========== Calculate metrics ==========%
%----- Calculation -----%
fprintf(1, '\nStarting:\n');

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

for k = 1 : length(gtImages)
    gtImage = cell2mat(gtImages(1,k));
    srImage = cell2mat(srImages(1,k));
    [rows_gt, columns_gt, ~] = size(gtImage);
    [rows_sr, columns_sr, ~] = size(srImage);
    scale = rows_sr/rows_gt;
    if (rows_gt ~= rows_sr && columns_gt ~= columns_sr)
        gtImage = imresize(gtImage, scale, "bicubic");
        gtImages(1,k) = {gtImage};
    end

    metricsFR(k).psnr = psnr(rgb2ycbcr(cell2mat(srImages(1,k))), rgb2ycbcr(cell2mat(gtImages(1,k))));
    metricsFR(k).ssim = ssim(cell2mat(srImages(1,k)), cell2mat(gtImages(1,k)));
    metricsFR(k).ms_ssim = mean(multissim(cell2mat(srImages(1,k)), cell2mat(gtImages(1,k))));
    metricsFR(k).fsim = fsim(cell2mat(srImages(1,k)), cell2mat(gtImages(1,k)));
end

fprintf(1, '\nFinished!\n');


%% ========== Calculate stats ==========%
%----- FR-metrics -----%
metricsFR_psnr.mean = mean([metricsFR.psnr]);
metricsFR_psnr.devstd = std([metricsFR.psnr]);
metricsFR_psnr.perc_inf = prctile([metricsFR.psnr], 5);
metricsFR_psnr.perc_sup = prctile([metricsFR.psnr], 95);

metricsFR_ssim.mean = mean([metricsFR.ssim]);
metricsFR_ssim.devstd = std([metricsFR.ssim]);
metricsFR_ssim.perc_inf = prctile([metricsFR.ssim], 5);
metricsFR_ssim.perc_sup = prctile([metricsFR.ssim], 95);

metricsFR_ms_ssim.mean = mean([metricsFR.ms_ssim]);
metricsFR_ms_ssim.devstd = std([metricsFR.ms_ssim]);
metricsFR_ms_ssim.perc_inf = prctile([metricsFR.ms_ssim], 5);
metricsFR_ms_ssim.perc_sup = prctile([metricsFR.ms_ssim], 95);

metricsFR_fsim.mean = mean([metricsFR.fsim]);
metricsFR_fsim.devstd = std([metricsFR.fsim]);
metricsFR_fsim.perc_inf = prctile([metricsFR.fsim], 5);
metricsFR_fsim.perc_sup = prctile([metricsFR.fsim], 95);

metricsFRstats = [metricsFR_psnr, metricsFR_ssim, metricsFR_ms_ssim, metricsFR_fsim];
metrics = ["psnr", "ssim", "ms_ssim", "fsim"];
for i = 1:length(metrics)
    metricsFRstats(i).metrics =  metrics(i);
end

%----- NR-metrics: GT -----%
gt_metricsNR_niqe.mean = mean([gt_metricsNR.niqe]);
gt_metricsNR_niqe.devstd = std([gt_metricsNR.niqe]);
gt_metricsNR_niqe.perc_inf = prctile([gt_metricsNR.niqe], 5);
gt_metricsNR_niqe.perc_sup = prctile([gt_metricsNR.niqe], 95);

gt_metricsNR_piqe.mean = mean([gt_metricsNR.piqe]);
gt_metricsNR_piqe.devstd = std([gt_metricsNR.piqe]);
gt_metricsNR_piqe.perc_inf = prctile([gt_metricsNR.piqe], 5);
gt_metricsNR_piqe.perc_sup = prctile([gt_metricsNR.piqe], 95);

gt_metricsNR_brisque.mean = mean([gt_metricsNR.brisque]);
gt_metricsNR_brisque.devstd = std([gt_metricsNR.brisque]);
gt_metricsNR_brisque.perc_inf = prctile([gt_metricsNR.brisque], 5);
gt_metricsNR_brisque.perc_sup = prctile([gt_metricsNR.brisque], 95);

% gt_metricsNR_ma.mean = mean([gt_metricsNR.ma]);
% gt_metricsNR_ma.devstd = std([gt_metricsNR.ma]);
% gt_metricsNR_ma.perc_inf = prctile([gt_metricsNR.ma], 5);
% gt_metricsNR_ma.perc_sup = prctile([gt_metricsNR.ma], 95);

gt_metricsNRstats = [gt_metricsNR_niqe, gt_metricsNR_piqe, gt_metricsNR_brisque];%, gt_metricsNR_ma];
metrics = ["niqe", "piqe", "brisque"];%, "ma"];
for i = 1:length(metrics)
    gt_metricsNRstats(i).metrics =  metrics(i);
end

%----- NR-metrics: SR -----%
sr_metricsNR_niqe.mean = mean([sr_metricsNR.niqe]);
sr_metricsNR_niqe.devstd = std([sr_metricsNR.niqe]);
sr_metricsNR_niqe.perc_inf = prctile([sr_metricsNR.niqe], 5);
sr_metricsNR_niqe.perc_sup = prctile([sr_metricsNR.niqe], 95);

sr_metricsNR_piqe.mean = mean([sr_metricsNR.piqe]);
sr_metricsNR_piqe.devstd = std([sr_metricsNR.piqe]);
sr_metricsNR_piqe.perc_inf = prctile([sr_metricsNR.piqe], 5);
sr_metricsNR_piqe.perc_sup = prctile([sr_metricsNR.piqe], 95);

sr_metricsNR_brisque.mean = mean([sr_metricsNR.brisque]);
sr_metricsNR_brisque.devstd = std([sr_metricsNR.brisque]);
sr_metricsNR_brisque.perc_inf = prctile([sr_metricsNR.brisque], 5);
sr_metricsNR_brisque.perc_sup = prctile([sr_metricsNR.brisque], 95);

% sr_metricsNR_ma.mean = mean([sr_metricsNR.ma]);
% sr_metricsNR_ma.devstd = std([sr_metricsNR.ma]);
% sr_metricsNR_ma.perc_inf = prctile([sr_metricsNR.ma], 5);
% sr_metricsNR_ma.perc_sup = prctile([sr_metricsNR.ma], 95);

sr_metricsNRstats = [sr_metricsNR_niqe, sr_metricsNR_piqe, sr_metricsNR_brisque];%, sr_metricsNR_ma];
metrics = ["niqe", "piqe", "brisque"];%, "ma"];
for i = 1:length(metrics)
    sr_metricsNRstats(i).metrics =  metrics(i);
end

timeElapsed = toc/60 %min

%% ========== Save files ==========%

%----- All metrics: -----%
% root_metrics = "D:/DOMI/University/Thesis/Results_resume/excel_metrics/";
% path_metrics = root_metrics + testSet_name;
%
% metricsFR_xls = path_metrics + "/metricsFR.xlsx";
% gt_metricsNR_xls = path_metrics + '/gt_metricsNR.xlsx';
% sr_metricsNR_xls = path_metrics + '/sr_metricsNR.xlsx';
%
% writetable(struct2table(metricsFR), metricsFR_xls);
% writetable(struct2table(gt_metricsNR), gt_metricsNR_xls);
% writetable(struct2table(sr_metricsNR), sr_metricsNR_xls);
%
% %----- All stats: -----%
% metricsFR_xls = path_metrics + "/stats_metricsFR.xlsx";
% gt_metricsNR_xls = path_metrics + '/stats_gt_metricsNR.xlsx';
% sr_metricsNR_xls = path_metrics + '/stats_sr_metricsNR.xlsx';
%
% writetable(struct2table(metricsFRstats), metricsFR_xls);
% writetable(struct2table(gt_metricsNRstats), gt_metricsNR_xls);
% writetable(struct2table(sr_metricsNRstats), sr_metricsNR_xls);


for k=1:10
    structTry(k).stats = sr_metricsNRstats;
    structTry(k).iteration = k;
end
